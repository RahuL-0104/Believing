"""
GROUNDING DINO ORCHESTRATOR

Runs Grounding DINO ONCE per image to detect the food item and the plate/
bowl, then sends those same two boxes to BOTH downstream models:
    - Model 1 (depth_model... no wait, model1_classifier / your classifier.py's
      FoodClassifierInference) -> dish label + confidence
    - Model 2 (depth_model.py's VolumeModel) -> estimated volume in cm^3

INSTALL:
    pip install torch torchvision opencv-python numpy --break-system-packages
    pip install groundingdino-py --break-system-packages
    (or: git clone https://github.com/IDEA-Research/GroundingDINO.git and
     pip install -e . from inside it, if the pip package above doesn't work
     cleanly on your setup - GroundingDINO's packaging has historically been
     a bit inconsistent across versions)

    Download the checkpoint + config from the GroundingDINO repo:
    https://github.com/IDEA-Research/GroundingDINO#luggage-checkpoints
    (groundingdino_swint_ogc.pth + its matching config .py file)

KNOWN LIMITATION (read before trusting volume outputs):
    Grounding DINO detects THAT a plate/bowl is present, but has no idea
    what its real-world diameter is - it only gives you a pixel box. Model 2
    needs a real cm diameter (via plate_type) to compute volume correctly.
    Right now this defaults to "dinner_plate" (26cm) for every detection,
    which will be WRONG whenever the real container is a different size.
    Fixing this properly means either (a) letting the user pick a plate
    size in your app's UI, or (b) adding a second detection step that
    estimates plate size from a reference object. Both are real follow-up
    work, not solved here - treat volume outputs as rough estimates until
    this is addressed.
"""

import os
import random
import cv2
import numpy as np
import torch

from groundingdino.util.inference import load_model, load_image, predict

from classifier import FoodClassifierInference     # your actual training script's built-in inference class
from depth_model import VolumeModel, PLATE_DIAMETERS_CM  # Model 2


# ---------------------------------------------------------------------------
# CONFIG - update these paths to match your actual setup
# ---------------------------------------------------------------------------
GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "checkpoints/groundingdino_swint_ogc.pth"

TEXT_PROMPT = "food . plate . bowl ."   # Grounding DINO phrase-grounding prompt
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

DEFAULT_PLATE_TYPE = "dinner_plate"   # used when we can't determine real size - see limitation note above


class FoodPipeline:
    def __init__(
        self,
        classifier: FoodClassifierInference,
        volume_model: VolumeModel,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dino_model = load_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_CHECKPOINT)
        self.dino_model = self.dino_model.to(self.device)

        self.classifier = classifier
        self.volume_model = volume_model
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    def _detect(self, image_path: str):
        """
        Runs Grounding DINO once. Returns the best food box and the best
        plate/bowl box, in pixel xyxy coordinates - or None for either if
        nothing matching was confidently detected.
        """
        image_source, image_tensor = load_image(image_path)
        h, w, _ = image_source.shape

        boxes, logits, phrases = predict(
            model=self.dino_model,
            image=image_tensor,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=self.device,
        )

        food_box, plate_box = None, None
        food_score, plate_score = -1, -1

        for box, score, phrase in zip(boxes, logits, phrases):
            # boxes come back normalized (cx, cy, w, h) in [0,1] - convert to
            # pixel xyxy
            cx, cy, bw, bh = box.tolist()
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            xyxy = [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]

            phrase_lower = phrase.lower()
            score_val = float(score)

            if "plate" in phrase_lower or "bowl" in phrase_lower:
                if score_val > plate_score:
                    plate_score = score_val
                    plate_box = xyxy
            else:
                # anything else matched by the prompt is treated as the food region
                if score_val > food_score:
                    food_score = score_val
                    food_box = xyxy

        return image_source, food_box, plate_box, food_score, plate_score

    # ------------------------------------------------------------------
    def process_image(self, image_path: str, plate_type: str = DEFAULT_PLATE_TYPE) -> dict:
        """
        Full pipeline on one image: detect -> classify -> estimate volume.
        Returns a dict with everything, or an "error" key explaining why a
        step was skipped (e.g. no food/plate detected).
        """
        image_rgb, food_box, plate_box, food_score, plate_score = self._detect(image_path)

        result = {
            "image_path": image_path,
            "food_box": food_box,
            "plate_box": plate_box,
            "food_detection_score": food_score,
            "plate_detection_score": plate_score,
        }

        if food_box is None:
            result["error"] = "No food region detected - skipping classification and volume."
            return result

        # ---- Model 1: classification (crop just the food box) ----
        x1, y1, x2, y2 = food_box
        food_crop = image_rgb[y1:y2, x1:x2]
        if food_crop.size == 0:
            result["error"] = "Food box was empty/degenerate - skipping classification and volume."
            return result

        classification = self.classifier.predict(food_crop, top_k=3)
        result["label"] = classification["label"]
        result["confidence"] = classification["confidence"]
        # FoodClassifierInference doesn't compute this itself - flagging low
        # confidence here instead, consistent with your earlier confirmation logic
        result["needs_confirmation"] = classification["confidence"] < self.confidence_threshold

        # ---- Model 2: volume (needs both food box and a plate box) ----
        if plate_box is None:
            result["volume_cm3"] = None
            result["volume_note"] = "No plate/bowl detected - volume not computed."
            return result

        try:
            volume_result = self.volume_model.estimate_volume(
                image_rgb=image_rgb,
                food_box_xyxy=food_box,
                plate_box_xyxy=plate_box,
                plate_type=plate_type,
            )
            result["volume_cm3"] = volume_result["volume_cm3"]
        except ValueError as e:
            result["volume_cm3"] = None
            result["volume_note"] = f"Volume estimation failed: {e}"

        return result


# ---------------------------------------------------------------------------
# TEST MODE - runs the full pipeline against real photos from your dataset
# and saves annotated images so you can VISUALLY verify the detections are
# correct, not just trust printed numbers.
# ---------------------------------------------------------------------------
def draw_result(image_rgb, result: dict, out_path: str):
    vis = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()

    if result.get("food_box"):
        x1, y1, x2, y2 = result["food_box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label_text = f"{result.get('label', '?')} ({result.get('confidence', 0):.2f})"
        cv2.putText(vis, label_text, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if result.get("plate_box"):
        x1, y1, x2, y2 = result["plate_box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)

    volume_text = f"volume: {result.get('volume_cm3', 'N/A')}"
    cv2.putText(vis, volume_text, (10, vis.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imwrite(out_path, vis)


def run_test(dataset_val_dir: str, num_samples: int = 10, output_dir: str = "pipeline_test_output"):
    """
    Samples a handful of real images across different classes from your val
    folder, runs the full pipeline on each, prints the results, and saves
    annotated images to output_dir so you can visually check the boxes and
    predictions look right.
    """
    os.makedirs(output_dir, exist_ok=True)

    class_folders = [
        d for d in os.listdir(dataset_val_dir)
        if os.path.isdir(os.path.join(dataset_val_dir, d))
    ]
    random.seed(42)
    sampled_classes = random.sample(class_folders, min(num_samples, len(class_folders)))

    classifier = FoodClassifierInference(
        checkpoint_path="outputs_efficientnet_v2m/best_model.pt",
    )
    volume_model = VolumeModel(
        sam2_checkpoint="checkpoints/sam2.1_hiera_small.pt",
        sam2_config="configs/sam2.1/sam2.1_hiera_s.yaml",
        device="cuda",
    )
    pipeline = FoodPipeline(classifier, volume_model, device="cuda")

    print(f"Testing pipeline on {len(sampled_classes)} sampled classes...\n")

    for cls in sampled_classes:
        cls_dir = os.path.join(dataset_val_dir, cls)
        images = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
        if not images:
            continue
        random.seed(0)
        img_name = random.choice(images)
        img_path = os.path.join(cls_dir, img_name)

        result = pipeline.process_image(img_path, plate_type=DEFAULT_PLATE_TYPE)

        print(f"--- true_class={cls} | file={img_name} ---")
        if "error" in result:
            print(f"  [ERROR] {result['error']}")
        else:
            correct = "OK" if result["label"] == cls else "MISMATCH"
            print(f"  predicted={result['label']} (conf={result['confidence']:.3f}) [{correct}]")
            print(f"  volume_cm3={result.get('volume_cm3')}")
            if result.get("volume_note"):
                print(f"  volume_note: {result['volume_note']}")

        image_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        out_path = os.path.join(output_dir, f"{cls}_{img_name}")
        draw_result(image_rgb, result, out_path)
        print(f"  saved annotated image to: {out_path}\n")

    print(f"\nDone. Check the '{output_dir}' folder to visually verify the boxes/labels/volumes.")


if __name__ == "__main__":
    # point this at your actual val folder to test against real data
    run_test(
        dataset_val_dir=r"C:\Users\sashank gowda\Desktop\Believing_model\Pre-processed_dataset_v2\val",
        num_samples=10,
    )
