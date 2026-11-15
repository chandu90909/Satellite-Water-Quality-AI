import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification.inference.quality_inference import (
    CLASS_NAMES,
    DEVICE,
    PATCH_SIZE,
    load_model,
    normalize_patch,
)
from common.normalization import normalize_bands
from common.ordering import sort_acquisitions
from common.patch_utils import pad_to_size
from preprocessing.utils.band_reader import read_required_bands


class GradCAM:
    def __init__(self, model, target_layer):
        self.activations = None
        self.gradients = None
        self.handle = target_layer.register_forward_hook(self._capture_activation)
        self.model = model

    def _capture_activation(self, _module, _inputs, output):
        self.activations = output
        output.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient):
        self.gradients = gradient

    def close(self):
        self.handle.remove()

    def __call__(self, tensor, class_index):
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None
        logits = self.model(tensor)
        logits[0, class_index].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        minimum = cam.min()
        maximum = cam.max()
        if float((maximum - minimum).detach()) <= 1e-8:
            return np.zeros(tuple(cam.shape), dtype=np.float32)
        cam = (cam - minimum) / (maximum - minimum)
        return cam.detach().cpu().numpy().astype(np.float32)


def display_rgb(patch):
    rgb = patch[..., [3, 1, 0]].astype(np.float32)
    for channel in range(rgb.shape[-1]):
        values = rgb[..., channel]
        low, high = np.percentile(values, (2, 98))
        if high <= low:
            rgb[..., channel] = 0.0
        else:
            rgb[..., channel] = np.clip((values - low) / (high - low), 0.0, 1.0)
    return rgb


def generate_for_image(lake_name, image_name, model):
    source_path = PROJECT_ROOT / "data" / "raw" / lake_name / image_name
    mask_path = PROJECT_ROOT / "data" / "masks" / "water" / lake_name / f"{Path(image_name).stem}_water_mask.tif"
    prediction_path = PROJECT_ROOT / "outputs" / "quality_predictions" / lake_name / f"{Path(image_name).stem}_quality_predictions.csv"
    if not source_path.exists() or not mask_path.exists() or not prediction_path.exists():
        raise FileNotFoundError(f"Missing source image, water mask, or prediction CSV for {lake_name}/{image_name}")

    predictions = pd.read_csv(prediction_path)
    if predictions.empty:
        print(f"No classified water patches for {lake_name}/{image_name}; skipping Grad-CAM.")
        return []

    predictions["confidence"] = pd.to_numeric(predictions["confidence"], errors="coerce")
    predictions = predictions[predictions["prediction"].isin(CLASS_NAMES)].dropna(subset=["confidence"])
    if predictions.empty:
        return []

    with rasterio.open(source_path) as source:
        bands = read_required_bands(source)
        image = np.stack([bands[name] for name in ("B3", "B4", "B5", "B8", "B11")], axis=-1)
        # Same scene-level stretch the classifier sees at inference.
        image = normalize_bands(image)
    with rasterio.open(mask_path) as mask_source:
        water_mask = mask_source.read(1).astype(np.uint8)

    if water_mask.shape[0] < image.shape[0] or water_mask.shape[1] < image.shape[1]:
        raise ValueError(f"Water-mask dimensions do not match source image: {lake_name}/{image_name}")
    water_mask = water_mask[: image.shape[0], : image.shape[1]]

    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)
    output_dir = PROJECT_ROOT / "outputs" / "gradcam" / lake_name
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    try:
        for class_name in CLASS_NAMES:
            class_rows = predictions[predictions["prediction"] == class_name]
            if class_rows.empty:
                continue
            selected = class_rows.loc[class_rows["confidence"].idxmax()]
            row, col = int(selected["row"]), int(selected["col"])
            patch = image[row:row + PATCH_SIZE, col:col + PATCH_SIZE]
            patch_mask = water_mask[row:row + PATCH_SIZE, col:col + PATCH_SIZE] > 0
            if patch.size == 0 or not patch_mask.any():
                continue

            # Edge tiles are smaller than the trained 128px input; pad them, then crop
            # the explanation back to the visible tile so no padded border is shown.
            visible_height, visible_width = patch.shape[0], patch.shape[1]
            padded_patch = pad_to_size(patch, PATCH_SIZE)

            tensor_array = normalize_patch(padded_patch)
            tensor = torch.from_numpy(tensor_array.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            class_index = CLASS_NAMES.index(class_name)
            with torch.enable_grad():
                cam = gradcam(tensor, class_index)

            cam = cam[:visible_height, :visible_width]
            cam[~patch_mask] = 0.0
            confidence = float(selected["confidence"])
            figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
            axes[0].imshow(display_rgb(patch))
            axes[0].imshow(~patch_mask, cmap="gray", alpha=0.35, vmin=0, vmax=1)
            axes[0].set_title("False-color input")
            axes[1].imshow(display_rgb(patch))
            axes[1].imshow(np.ma.masked_where(~patch_mask, cam), cmap="magma", alpha=0.62, vmin=0, vmax=1)
            axes[1].set_title(f"{class_name} activation | confidence {confidence:.3f}")
            for axis in axes:
                axis.set_axis_off()

            output_path = output_dir / f"{Path(image_name).stem}_{class_name.lower()}_gradcam.png"
            figure.savefig(output_path, dpi=160)
            plt.close(figure)
            saved_paths.append(output_path)
            print(f"Saved {output_path}")
    finally:
        gradcam.close()

    return saved_paths


def generate_for_lake(lake_name, image_names=None, model=None, log=print):
    """Generate Grad-CAM overlays for every classified acquisition of a lake."""
    if image_names is None:
        lake_dir = PROJECT_ROOT / "data" / "raw" / lake_name
        image_names = [path.name for path in sort_acquisitions(lake_dir.glob("*.tif"))]
    model = model or load_model()
    saved = []
    for image_name in image_names:
        prediction_path = (
            PROJECT_ROOT / "outputs" / "quality_predictions" / lake_name
            / f"{Path(image_name).stem}_quality_predictions.csv"
        )
        if not prediction_path.exists():
            log(f"No predictions for {lake_name}/{image_name}; skipping Grad-CAM")
            continue
        saved.extend(generate_for_image(lake_name, image_name, model))
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate class-specific Grad-CAM overlays from existing quality predictions.")
    parser.add_argument("--lake", type=str, help="Lake name, e.g. Hussain_Sagar")
    parser.add_argument("--image", type=str, help="Image name, e.g. Hussain_Sagar_1.tif")
    parser.add_argument("--all", action="store_true", help="Generate Grad-CAM for every image with prediction outputs")
    args = parser.parse_args(argv)

    if not args.all and not args.lake:
        parser.error("Provide --lake and optional --image, or use --all")
    if args.image and not args.lake:
        parser.error("--image requires --lake")

    model = load_model()
    if args.all:
        items = [
            (lake_dir.name, image_path.name)
            for lake_dir in sorted((PROJECT_ROOT / "data" / "raw").iterdir())
            if lake_dir.is_dir()
            for image_path in sort_acquisitions(lake_dir.glob("*.tif"))
            if (PROJECT_ROOT / "outputs" / "quality_predictions" / lake_dir.name / f"{image_path.stem}_quality_predictions.csv").exists()
        ]
    elif args.image:
        items = [(args.lake, args.image)]
    else:
        items = [
            (args.lake, image_path.name)
            for image_path in sort_acquisitions((PROJECT_ROOT / "data" / "raw" / args.lake).glob("*.tif"))
        ]

    generated = 0
    for lake_name, image_name in items:
        generated += len(generate_for_image(lake_name, image_name, model))
    print(f"Generated {generated} Grad-CAM image(s) across {len(items)} source image(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
