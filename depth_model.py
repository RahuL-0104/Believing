"""
MODEL 2 — Volume Estimator (SAM2 + Depth Anything V2)

Takes bounding boxes from Grounding DINO (food box + plate/bowl box) and
returns an estimated volume in cm^3.

PIPELINE INSIDE THIS FILE:
    1. SAM2 turns each box into a precise pixel mask (food mask, plate mask).
    2. Depth Anything V2 gives a RELATIVE depth map for the whole image
       (monocular depth has no inherent real-world scale).
    3. The plate's known real-world diameter converts pixel distances to cm.
    4. Food height above the plate surface, integrated across every masked
       pixel's footprint area, gives volume.

ACCURACY CAVEAT (read once, keep in mind always):
    This is a monocular single-image estimate, not a lab measurement. It only
    sees the visible top surface — it can't know what's hidden under
    overhangs, inside deep bowls, or under garnish. Two things matter more
    than model choice for real accuracy:
      - DEPTH_SCALE_K below MUST be calibrated against real known-volume
        photos before you trust any output number. It ships at 1.0, which is
        almost certainly wrong for your setup.
      - Consistent camera angle across user photos (guided ~45-60 degrees
        from above works best) matters more than any model swap.

INSTALL:
    pip install torch torchvision opencv-python numpy transformers --break-system-packages
    pip install git+https://github.com/facebookresearch/segment-anything-2.git --break-system-packages
    Download a SAM2 checkpoint (sam2_hiera_small or sam2_hiera_base_plus
    recommended for 8GB laptop VRAM budgets, alongside Model 1 + Grounding DINO
    running in the same process) from:
    https://github.com/facebookresearch/segment-anything-2
"""

import numpy as np
import cv2
import torch
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2


# ---------------------------------------------------------------------------
# Known standard plate/bowl diameters (cm) — used as the metric scale
# reference. Extend this list with whatever vessel types your Grounding DINO
# stage is prompted to detect, or whatever your app lets the user select.
# ---------------------------------------------------------------------------
PLATE_DIAMETERS_CM = {
    "small_plate": 18.0,
    "dinner_plate": 26.0,
    "thali_plate": 28.0,
    "bowl_small": 12.0,
    "bowl_medium": 15.0,
    "bowl_large": 18.0,
    "steel_plate_25_5cm": 25.5,    # calibration container - measured with ruler
    "steel_bowl_12_6cm": 12.6,     # calibration container - measured with ruler
    "steel_tumbler_7cm": 7.0,      # calibration container - measured with ruler
    "green_jug_8_7cm": 8.7,        # calibration container - measured with ruler
    "glass_tumbler_8_1cm": 8.1,    # calibration container - measured with ruler
    "glass_wine_6cm": 6.0,          # calibration container - measured with ruler
}

# Calibration constant converting Depth Anything V2's relative depth units
# into real-world centimeters. THIS MUST BE TUNED against a small set of
# known-volume reference photos before trusting outputs - see
# calibrate_depth_scale.py. 1.0 is a placeholder, not a real value.
#
# Why this is safe to calibrate as a single multiplier: volume is LINEAR in
# DEPTH_SCALE_K (height_cm scales directly with it, and volume is a sum of
# height_cm x pixel_area). So K can be solved with simple least-squares
# across a handful of known-volume photos - see the calibration script.
# Calibrated 2026-08-28 from 5 real-world samples (turmeric water in a
# 6cm wine glass at 100/200/300ml, a 25.5cm steel plate at 300ml, and an
# 8.7cm jug at 300ml). Individual fits across different sample subsets
# ranged ~2700-3200; 3000 was chosen as the representative value.
#
# KNOWN LIMITATION: the depth model (Depth Anything V2 Small) only produced
# a real, non-trivial height reading for the highest-volume sample (300ml) -
# the 100ml/200ml readings were both near-zero, suggesting a sensitivity
# floor for shallow liquid layers. Expect this pipeline to be least accurate
# for thin/shallow food and more reliable for mounded, higher-volume food.
# Recalibrate with more samples (especially varied heights on the SAME
# reliable tall/narrow container shape) if better precision is needed later.
DEPTH_SCALE_K = 3000.0


class VolumeModel:
    def __init__(
        self,
        sam2_checkpoint: str,
        sam2_config: str,
        depth_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str = "cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"

        # --- SAM2 ---
        sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # --- Depth Anything V2 ---
        self.depth_processor = AutoImageProcessor.from_pretrained(depth_model_id)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_id).to(self.device)
        self.depth_model.eval()

    # ------------------------------------------------------------------
    def _get_mask(self, image_rgb: np.ndarray, box_xyxy: list) -> np.ndarray:
        """SAM2: box -> precise pixel mask."""
        self.sam2_predictor.set_image(image_rgb)
        masks, scores, _ = self.sam2_predictor.predict(
            box=np.array(box_xyxy),
            multimask_output=False,
        )
        return masks[0].astype(bool)

    def _get_relative_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        """Depth Anything V2: image -> relative depth map (larger = closer)."""
        inputs = self.depth_processor(images=image_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        depth_map = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(image_rgb.shape[0], image_rgb.shape[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        return depth_map.astype(np.float32)

    def _plate_pixel_diameter(self, plate_mask: np.ndarray) -> float:
        """
        Fits an ellipse to the plate's mask contour and returns the MAJOR
        axis length in pixels. A circular plate photographed at an angle
        projects as an ellipse, not a circle - its bounding-box width/height
        is NOT its true diameter (it's foreshortened on one axis). The major
        axis of the fitted ellipse is the closest single-image approximation
        to the plate's true diameter regardless of viewing angle.
        """
        mask_u8 = (plate_mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise ValueError("Empty plate mask - plate not detected correctly.")

        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) < 5:
            # fitEllipse needs at least 5 points - fall back to bbox if the
            # mask is too small/thin to fit an ellipse reliably
            ys, xs = np.where(plate_mask)
            return float(max(xs.max() - xs.min(), ys.max() - ys.min()))

        (center, (minor_axis, major_axis), angle) = cv2.fitEllipse(largest_contour)
        return float(max(minor_axis, major_axis))

    # ------------------------------------------------------------------
    def estimate_volume(
        self,
        image_rgb: np.ndarray,
        food_box_xyxy: list,
        plate_box_xyxy: list,
        plate_type: str = "dinner_plate",
    ) -> dict:
        """
        image_rgb      : HxWx3 numpy array, RGB, the FULL original photo
                          (not pre-cropped - SAM2/Depth Anything need context)
        food_box_xyxy  : [x1,y1,x2,y2] from Grounding DINO, food item
        plate_box_xyxy : [x1,y1,x2,y2] from Grounding DINO, plate/bowl
        plate_type     : key into PLATE_DIAMETERS_CM

        Returns:
            {
              "volume_cm3": float,
              "mean_food_height_cm": float,
              "max_food_height_cm": float,
              "px_per_cm": float,
              "food_pixel_count": int,
            }
        """
        if plate_type not in PLATE_DIAMETERS_CM:
            raise ValueError(f"Unknown plate_type '{plate_type}'. Options: {list(PLATE_DIAMETERS_CM)}")

        food_mask = self._get_mask(image_rgb, food_box_xyxy)
        plate_mask = self._get_mask(image_rgb, plate_box_xyxy)
        depth_map = self._get_relative_depth(image_rgb)

        # Erode both masks by a couple pixels - SAM2 masks often bleed
        # slightly into background at object boundaries, and those edge
        # pixels get counted as "food" incorrectly, skewing volume.
        erosion_kernel = np.ones((3, 3), np.uint8)
        food_mask = cv2.erode(food_mask.astype(np.uint8), erosion_kernel, iterations=1).astype(bool)
        plate_mask = cv2.erode(plate_mask.astype(np.uint8), erosion_kernel, iterations=1).astype(bool)

        real_plate_diameter_cm = PLATE_DIAMETERS_CM[plate_type]
        plate_pixel_diameter = self._plate_pixel_diameter(plate_mask)

        px_per_cm = plate_pixel_diameter / real_plate_diameter_cm
        cm_per_px = 1.0 / px_per_cm
        pixel_area_cm2 = cm_per_px ** 2

        # "zero height" baseline: robust (trimmed) plate surface depth,
        # excluding wherever the food itself sits. Using a percentile-trimmed
        # median instead of a raw median further reduces sensitivity to
        # reflections/glare on the plate surface.
        plate_ref_depths = depth_map[plate_mask & ~food_mask]
        if len(plate_ref_depths) == 0:
            raise ValueError("No visible plate surface outside the food mask - plate may be fully covered by food.")
        lo, hi = np.percentile(plate_ref_depths, [5, 95])
        trimmed_plate_depths = plate_ref_depths[(plate_ref_depths >= lo) & (plate_ref_depths <= hi)]
        plate_surface_depth = float(np.median(trimmed_plate_depths)) if len(trimmed_plate_depths) else float(np.median(plate_ref_depths))

        food_depth_values = depth_map[food_mask]
        # Clip extreme values (steam, garnish spikes, single-pixel depth
        # noise) before summing - volume is a SUM over all food pixels, so a
        # handful of outlier pixels can meaningfully distort the total.
        if len(food_depth_values) > 0:
            lo_f, hi_f = np.percentile(food_depth_values, [1, 99])
            food_depth_values = np.clip(food_depth_values, lo_f, hi_f)

        height_relative = np.clip(food_depth_values - plate_surface_depth, a_min=0, a_max=None)
        height_cm = height_relative * cm_per_px * DEPTH_SCALE_K

        voxel_volumes = height_cm * pixel_area_cm2
        total_volume_cm3 = float(np.sum(voxel_volumes))

        return {
            "volume_cm3": total_volume_cm3,
            "mean_food_height_cm": float(np.mean(height_cm)) if len(height_cm) else 0.0,
            "max_food_height_cm": float(np.max(height_cm)) if len(height_cm) else 0.0,
            "px_per_cm": px_per_cm,
            "food_pixel_count": int(food_mask.sum()),
        }


if __name__ == "__main__":
    # Running this file directly does a quick self-test on ONE of your
    # calibration photos, so you can visibly confirm Model 2 works end to
    # end - without this block, running "python depth_model.py" does
    # nothing and exits instantly, because the rest of this file only
    # DEFINES the model/class - it doesn't run anything on its own.
    import cv2

    print("Running Model 2 self-test on a real photo...")

    model = VolumeModel(
        sam2_checkpoint="checkpoints/sam2.1_hiera_small.pt",
        sam2_config="configs/sam2.1/sam2.1_hiera_s.yaml",
        device="cuda",
    )

    # using your known-good 300ml wine glass photo as the test image
    test_image_path = "calibrate_photos/sample6_glass_juice_300ml.jpg"
    image_bgr = cv2.imread(test_image_path)
    if image_bgr is None:
        print(f"Could not read test image: {test_image_path}")
        print("Update test_image_path in this file to point at a real photo you have.")
    else:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = model.estimate_volume(
            image_rgb=image_rgb,
            food_box_xyxy=[160, 298, 497, 720],
            plate_box_xyxy=[150, 280, 505, 720],
            plate_type="glass_wine_6cm",
        )
        print("\n" + "=" * 40)
        print(f"Estimated volume: {result['volume_cm3']:.1f} cm^3")
        print(f"(known real volume for this photo: 300 cm^3)")
        print(f"Mean food height: {result['mean_food_height_cm']:.2f} cm")
        print(f"Max food height:  {result['max_food_height_cm']:.2f} cm")
        print("=" * 40)
