import os
import json
import math
import time
import copy
import random
import zipfile
import warnings
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_V2_M_Weights

warnings.filterwarnings("ignore")


@dataclass
class Config:
    data_path: str = r"C:\Users\sashank gowda\Desktop\Believing_model\Pre-processed_dataset_v2"
    output_dir: str = "outputs_efficientnet_v2m"
    extract_dir: Optional[str] = None

    image_size: int = 384
    batch_size: int = 4          # laptop 4060 has less usable VRAM than desktop
                                   # (Windows/display + laptop TDP limits) - safer default
    num_workers: int = 2
    grad_accum_steps: int = 8    # doubled to keep effective batch size the same (32)

    gradient_checkpointing: bool = True  # trades ~20-30% slower training for
                                           # significantly less VRAM - needed
                                           # for V2-M at 384px on 8GB laptops

    head_only_epochs: int = 2
    finetune_epochs: int = 18
    lr_head: float = 1e-3
    lr_finetune: float = 2e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    dropout: float = 0.3
    early_stopping_patience: int = 6

    amp: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    save_confusion_matrix: bool = True
    save_every_epoch: bool = False
    use_weighted_loss: bool = True   # ON by default - you have classes ranging 12 to 7744 images
    max_class_weight_ratio: float = 10.0  # caps extreme weights so thin classes don't destabilize training

    resume_from: Optional[str] = None  # path to a "last_checkpoint.pt" to resume an interrupted run


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # free speed on RTX 40-series (Ada) with no accuracy cost - TF32 matmuls
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_dataset_root(data_path: str, extract_dir: Optional[str] = None) -> Path:
    data_path = Path(data_path)
    if data_path.is_dir():
        return data_path

    if data_path.suffix.lower() != ".zip":
        raise ValueError("data_path must point to either an unzipped dataset directory or a .zip file")

    target_root = Path(extract_dir) if extract_dir else data_path.with_suffix("")
    marker = target_root / ".extracted_ok"
    if marker.exists() and target_root.exists():
        return target_root

    ensure_dir(target_root)
    print(f"Extracting dataset from: {data_path}")
    with zipfile.ZipFile(data_path, "r") as zf:
        zf.extractall(target_root)
    marker.write_text("ok")

    children = [p for p in target_root.iterdir() if p.is_dir()]
    if len(children) == 1 and all((children[0] / split).exists() for split in ["train", "val"]):
        return children[0]
    return target_root


def clean_empty_classes(dataset_root: Path, splits=("train", "val", "test")) -> None:
    """Remove any class folder that has zero images in ANY split - prevents
    ImageFolder / training crashes on empty classes left over from merges
    or failed scraping."""
    import shutil

    empty_classes = set()
    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue
        for cls_dir in split_dir.iterdir():
            if cls_dir.is_dir():
                file_count = sum(1 for f in cls_dir.iterdir() if f.is_file())
                if file_count == 0:
                    empty_classes.add(cls_dir.name)

    if not empty_classes:
        print("No empty class folders found.")
        return

    for cls in sorted(empty_classes):
        for split in splits:
            cls_path = dataset_root / split / cls
            if cls_path.is_dir():
                shutil.rmtree(cls_path)
    print(f"Removed {len(empty_classes)} empty class folders: {sorted(empty_classes)}")


def validate_dataset_structure(dataset_root: Path) -> None:
    required = [dataset_root / "train", dataset_root / "val"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset structure not found. Expected at least these folders: "
            f"{missing}. Root should contain train/ and val/."
        )


def get_transforms(image_size: int):
    weights = EfficientNet_V2_M_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_tfms, eval_tfms


def make_dataloaders(cfg: Config):
    dataset_root = resolve_dataset_root(cfg.data_path, cfg.extract_dir)
    validate_dataset_structure(dataset_root)
    clean_empty_classes(dataset_root)

    train_tfms, eval_tfms = get_transforms(cfg.image_size)
    train_ds = datasets.ImageFolder(dataset_root / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(dataset_root / "val", transform=eval_tfms)
    test_ds = datasets.ImageFolder(dataset_root / "test", transform=eval_tfms) if (dataset_root / "test").exists() else None

    common_loader_args = dict(
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **common_loader_args)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **common_loader_args)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **common_loader_args) if test_ds else None

    idx_to_class = {idx: name for name, idx in train_ds.class_to_idx.items()}
    class_to_idx = train_ds.class_to_idx

    class_weights = None
    if cfg.use_weighted_loss:
        counts = np.bincount(train_ds.targets)
        weights = counts.sum() / np.maximum(counts, 1)
        weights = weights / weights.mean()
        # cap extreme weights - without this, a 12-image class can get a
        # weight 600x+ larger than a well-represented class, which makes
        # training unstable (huge gradient spikes on tiny classes) rather
        # than actually helping them
        weights = np.clip(weights, a_min=None, a_max=cfg.max_class_weight_ratio)
        class_weights = torch.tensor(weights, dtype=torch.float32)

    return train_loader, val_loader, test_loader, class_to_idx, idx_to_class, class_weights


class EfficientNetV2MClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3, gradient_checkpointing: bool = False):
        super().__init__()
        weights = EfficientNet_V2_M_Weights.DEFAULT
        self.backbone = models.efficientnet_v2_m(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            # Splits backbone.features (an nn.Sequential of stages) into
            # segments and checkpoints them: activations aren't stored for
            # backward, they're recomputed on the fly instead. This is what
            # actually cuts peak VRAM - the earlier version of this method
            # didn't do real checkpointing, this one does.
            x = torch.utils.checkpoint.checkpoint_sequential(
                self.backbone.features, segments=4, input=x, use_reentrant=False
            )
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.backbone.classifier(x)
            return x
        return self.backbone(x)


def freeze_backbone(model: EfficientNetV2MClassifier, freeze: bool = True) -> None:
    for param in model.backbone.features.parameters():
        param.requires_grad = not freeze


class EarlyStopping:
    def __init__(self, patience: int = 6):
        self.patience = patience
        self.best = -float("inf")
        self.bad_epochs = 0

    def step(self, score: float) -> bool:
        if score > self.best:
            self.best = score
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 5)) -> List[float]:
    with torch.no_grad():
        max_k = min(max(ks), logits.size(1))
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        scores = []
        for k in ks:
            k = min(k, logits.size(1))
            correct_k = correct[:k].reshape(-1).float().sum(0)
            scores.append((correct_k / targets.size(0)).item())
        return scores


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: str,
    epoch: int,
    total_epochs: int,
    grad_accum_steps: int = 1,
    scheduler=None,
    train: bool = True,
) -> Dict[str, float]:
    model.train(train)
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    if train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    desc = f"Epoch {epoch}/{total_epochs} | {'train' if train else 'val'}"
    total_steps = len(loader)
    print(f"{desc} - starting ({total_steps} steps)...")

    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            use_amp = bool(device.startswith("cuda")) and scaler is not None
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
                if train:
                    loss = loss / grad_accum_steps

            if train:
                scaler.scale(loss).backward()
                if step % grad_accum_steps == 0 or step == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

        batch_loss = loss.item() * (grad_accum_steps if train else 1.0)
        top1, top5 = topk_accuracy(logits, targets, ks=(1, 5))
        preds = torch.argmax(logits, dim=1)

        bs = images.size(0)
        total_samples += bs
        running_loss += batch_loss * bs
        running_top1 += top1 * bs
        running_top5 += top5 * bs
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

    # single summary line per completed epoch/phase - no live per-batch bar,
    # avoids the redraw/duplicate-line issue some terminals show with tqdm
    print(
        f"{desc} - done | loss={running_loss / total_samples:.4f} "
        f"top1={running_top1 / total_samples:.4f} top5={running_top5 / total_samples:.4f}"
    )

    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    return {
        "loss": running_loss / max(total_samples, 1),
        "top1": running_top1 / max(total_samples, 1),
        "top5": running_top5 / max(total_samples, 1),
        "macro_f1": macro_f1,
        "preds": all_preds,
        "targets": all_targets,
    }


def save_json(data: Dict, path: Union[str, Path]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: Config,
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[int, str],
    best_metric: float,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(cfg),
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
        "best_metric": best_metric,
    }
    torch.save(checkpoint, path)


def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    output_dir = ensure_dir(cfg.output_dir)

    print(f"Device: {cfg.device}")
    print("Recommended input for training is an **unzipped dataset path**.")
    print("This script can also accept a .zip path and extract it once automatically.")

    train_loader, val_loader, test_loader, class_to_idx, idx_to_class, class_weights = make_dataloaders(cfg)
    num_classes = len(class_to_idx)

    save_json(class_to_idx, output_dir / "class_to_idx.json")
    save_json({str(k): v for k, v in idx_to_class.items()}, output_dir / "idx_to_class.json")
    save_json(asdict(cfg), output_dir / "train_config.json")

    model = EfficientNetV2MClassifier(
        num_classes=num_classes,
        dropout=cfg.dropout,
        gradient_checkpointing=cfg.gradient_checkpointing,
    ).to(cfg.device)
    if class_weights is not None:
        class_weights = class_weights.to(cfg.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp and cfg.device.startswith("cuda")))

    total_epochs = cfg.head_only_epochs + cfg.finetune_epochs
    best_score = -float("inf")
    best_epoch = 0
    best_weights = None
    stopper = EarlyStopping(patience=cfg.early_stopping_patience)

    start_time = time.time()
    global_epoch = 0

    # ---- RESUME SUPPORT ----
    # If cfg.resume_from points at a checkpoint, reload model weights,
    # best_score, and where we left off, so a crash mid-run (or an
    # intentional stop) doesn't cost you the whole run.
    epochs_already_done = 0
    resume_ckpt = None
    if cfg.resume_from and Path(cfg.resume_from).exists():
        print(f"\nResuming from checkpoint: {cfg.resume_from}")
        resume_ckpt = torch.load(cfg.resume_from, map_location=cfg.device)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        epochs_already_done = resume_ckpt.get("epoch", 0)
        best_score = resume_ckpt.get("best_metric", -float("inf"))
        best_weights = copy.deepcopy(model.state_dict())
        print(f"Resumed: {epochs_already_done} epochs already done, best_score so far = {best_score:.4f}")

    optimizer_restored = False  # only attempt the optimizer-state restore once,
                                  # at the first phase where real training resumes

    freeze_backbone(model, freeze=True)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, math.ceil(len(train_loader) / cfg.grad_accum_steps) * cfg.head_only_epochs))

    for phase_name, phase_epochs, phase_lr, freeze in [
        ("head", cfg.head_only_epochs, cfg.lr_head, True),
        ("finetune", cfg.finetune_epochs, cfg.lr_finetune, False),
    ]:
        if phase_epochs <= 0:
            continue

        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()  # release the previous phase's optimizer state before allocating a new one

        freeze_backbone(model, freeze=freeze)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=phase_lr, weight_decay=cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, math.ceil(len(train_loader) / cfg.grad_accum_steps) * phase_epochs))

        # If this is the phase where a resumed run actually starts training
        # again, try to restore the optimizer's momentum/adaptive-LR state
        # too, not just the model weights. This only works cleanly if the
        # checkpoint's optimizer was saved for the SAME set of trainable
        # parameters (i.e. resuming mid-phase, not switching phases) - if the
        # param groups don't match (e.g. resuming into finetune from a
        # checkpoint saved during head-only phase, where fewer params were
        # trainable), it's safely skipped rather than crashing the run.
        if resume_ckpt is not None and not optimizer_restored and epochs_already_done < (
            (cfg.head_only_epochs if phase_name == "head" else cfg.head_only_epochs + cfg.finetune_epochs)
        ):
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
                print("Restored optimizer state (momentum/adaptive LR) from checkpoint.")
            except (ValueError, KeyError) as e:
                print(f"Could not restore optimizer state (likely a phase change) - starting this "
                      f"optimizer fresh instead. This causes a brief, temporary accuracy dip while "
                      f"Adam's internal statistics rebuild, not a real regression. Detail: {e}")
            optimizer_restored = True

        print(f"\nStarting phase: {phase_name} | epochs={phase_epochs} | lr={phase_lr} | freeze_backbone={freeze}")

        for _ in range(phase_epochs):
            # if resuming past this epoch's position, skip straight through
            # without retraining it - just keep the counter in sync
            if global_epoch < epochs_already_done:
                global_epoch += 1
                continue

            global_epoch += 1
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=cfg.device,
                epoch=global_epoch,
                total_epochs=total_epochs,
                grad_accum_steps=cfg.grad_accum_steps,
                scheduler=scheduler,
                train=True,
            )

            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=None,
                scaler=None,
                device=cfg.device,
                epoch=global_epoch,
                total_epochs=total_epochs,
                grad_accum_steps=1,
                scheduler=None,
                train=False,
            )

            score = 0.7 * val_metrics["top1"] + 0.3 * val_metrics["macro_f1"]
            print(
                f"Epoch {global_epoch}/{total_epochs} | "
                f"train_loss={train_metrics['loss']:.4f} train_top1={train_metrics['top1']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1']:.4f} "
                f"val_top5={val_metrics['top5']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
            )

            if score > best_score:
                best_score = score
                best_epoch = global_epoch
                best_weights = copy.deepcopy(model.state_dict())
                save_checkpoint(
                    output_dir / "best_model.pt",
                    model,
                    optimizer,
                    global_epoch,
                    cfg,
                    class_to_idx,
                    idx_to_class,
                    best_score,
                )
                print(f"Saved new best checkpoint at epoch {global_epoch}")

            # always save a "last_checkpoint" so a crash never costs more
            # than one epoch of progress - this is separate from best_model.pt
            save_checkpoint(
                output_dir / "last_checkpoint.pt",
                model,
                optimizer,
                global_epoch,
                cfg,
                class_to_idx,
                idx_to_class,
                best_score,
            )

            if cfg.save_every_epoch:
                save_checkpoint(
                    output_dir / f"epoch_{global_epoch}.pt",
                    model,
                    optimizer,
                    global_epoch,
                    cfg,
                    class_to_idx,
                    idx_to_class,
                    best_score,
                )

            if stopper.step(score):
                print(f"Early stopping triggered at epoch {global_epoch}")
                break

        if stopper.bad_epochs >= cfg.early_stopping_patience:
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed / 60:.2f} minutes. Best epoch: {best_epoch}")

    final_val = run_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        optimizer=None,
        scaler=None,
        device=cfg.device,
        epoch=best_epoch,
        total_epochs=total_epochs,
        train=False,
    )

    report = classification_report(
        final_val["targets"],
        final_val["preds"],
        target_names=[idx_to_class[i] for i in range(len(idx_to_class))],
        output_dict=True,
        zero_division=0,
    )
    save_json(report, output_dir / "validation_classification_report.json")

    if cfg.save_confusion_matrix:
        cm = confusion_matrix(final_val["targets"], final_val["preds"])
        np.save(output_dir / "validation_confusion_matrix.npy", cm)

    if test_loader is not None:
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            optimizer=None,
            scaler=None,
            device=cfg.device,
            epoch=best_epoch,
            total_epochs=total_epochs,
            train=False,
        )
        test_summary = {
            "test_loss": test_metrics["loss"],
            "test_top1": test_metrics["top1"],
            "test_top5": test_metrics["top5"],
            "test_macro_f1": test_metrics["macro_f1"],
        }
        save_json(test_summary, output_dir / "test_metrics.json")
        print(f"Test top1={test_metrics['top1']:.4f} | test_top5={test_metrics['top5']:.4f} | test_macro_f1={test_metrics['macro_f1']:.4f}")


class FoodClassifierInference:
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        cfg_dict = ckpt.get("config", {})
        self.image_size = int(cfg_dict.get("image_size", 384))
        idx_to_class_raw = ckpt["idx_to_class"]
        self.idx_to_class = {int(k): v for k, v in idx_to_class_raw.items()}

        self.model = EfficientNetV2MClassifier(
            num_classes=len(self.idx_to_class),
            dropout=float(cfg_dict.get("dropout", 0.3)),
            gradient_checkpointing=False,  # never needed at inference - no backward pass
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        _, self.eval_tfms = get_transforms(self.image_size)

    def _to_pil(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError("image must be a file path, PIL.Image, or numpy array")

    @torch.inference_mode()
    def predict(self, image: Union[str, Path, Image.Image, np.ndarray], top_k: int = 3) -> Dict:
        image = self._to_pil(image)
        x = self.eval_tfms(image).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        confs, indices = torch.topk(probs, k=min(top_k, probs.shape[1]), dim=1)

        top_predictions = []
        for conf, idx in zip(confs[0].tolist(), indices[0].tolist()):
            top_predictions.append({
                "label": self.idx_to_class[idx],
                "confidence": round(float(conf), 4),
            })

        return {
            "label": top_predictions[0]["label"],
            "confidence": top_predictions[0]["confidence"],
            "top_k": top_predictions,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNetV2-M food classifier")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a last_checkpoint.pt to resume an interrupted run. "
             "Omit this flag to start a fresh run from scratch."
    )
    args = parser.parse_args()

    cfg = Config(
        data_path=r"C:\Users\sashank gowda\Desktop\Believing_model\Pre-processed_dataset_v2",
        output_dir="outputs_efficientnet_v2m",
        extract_dir=None,
        image_size=384,
        batch_size=4,
        num_workers=2,
        grad_accum_steps=8,
        gradient_checkpointing=True,
        head_only_epochs=2,
        finetune_epochs=18,
        lr_head=1e-3,
        lr_finetune=2e-4,
        weight_decay=1e-4,
        label_smoothing=0.1,
        dropout=0.3,
        early_stopping_patience=6,
        amp=True,
        seed=42,
        use_weighted_loss=True,
        max_class_weight_ratio=10.0,
        resume_from=args.resume,   # None if --resume wasn't passed - starts fresh
    )

    if cfg.resume_from:
        print(f"Resume flag set - will continue from: {cfg.resume_from}")
    else:
        print("No --resume flag passed - starting a fresh training run.")

    train(cfg)
