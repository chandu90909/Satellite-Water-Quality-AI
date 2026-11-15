import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torchvision.models import efficientnet_b0

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.normalization import normalize_bands
from common.ordering import sort_acquisitions
from common.patch_utils import pad_to_size
from preprocessing.utils.band_reader import read_required_bands

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 128
CLASS_NAMES = ["Good", "Moderate", "Poor"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SEVERITY_MAP = {"Good": "Low", "Moderate": "Medium", "Poor": "High"}
CSV_COLUMNS = ["lake", "image", "row", "col", "prediction", "confidence", "severity"]


def load_model(model_path: Path = None):
    model_path = model_path or PROJECT_ROOT / "models" / "efficientnet" / "efficientnet_b0_best.pth"
    checkpoint = torch.load(str(model_path), map_location=DEVICE)
    model = efficientnet_b0(weights=None)
    model.features[0][0] = torch.nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1, bias=False)
    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=3, bias=True)
    report = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if report.missing_keys or report.unexpected_keys:
        print(
            f"Warning: {model_path.name} did not match the architecture exactly "
            f"(missing={report.missing_keys}, unexpected={report.unexpected_keys})",
            file=sys.stderr,
        )
    model.to(DEVICE)
    model.eval()
    return model


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """Sanitize an already normalized patch before it enters the network.

    Scene-level reflectance is stretched with :func:`common.normalization.normalize_bands`
    before tiling; this guard only removes non-finite values and keeps the
    tensor inside the range the classifier was trained on.
    """
    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
    patch = np.clip(patch, 0.0, 1.0)
    return patch.astype(np.float32)


def extract_patch_candidates(image_array: np.ndarray, mask_array: np.ndarray):
    height, width = mask_array.shape
    candidates = []
    for row in range(0, height, PATCH_SIZE):
        for col in range(0, width, PATCH_SIZE):
            patch_mask = mask_array[row: row + PATCH_SIZE, col: col + PATCH_SIZE]
            if patch_mask.size == 0:
                continue
            if np.any(patch_mask):
                candidates.append((row, col))
    return candidates


def predict_single_patch(model, patch_array: np.ndarray):
    if patch_array.ndim != 3 or patch_array.shape[2] != 5:
        raise ValueError(f"Expected a 5-band patch, got shape {patch_array.shape}")
    patch = normalize_patch(pad_to_size(patch_array, PATCH_SIZE))
    tensor = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        prob, pred_idx = torch.max(probs, dim=1)
    pred_idx = int(pred_idx.item())
    confidence = float(prob.item())
    label = CLASS_NAMES[pred_idx]
    return label, confidence, pred_idx


def process_image(lake_name: str, image_name: str, model=None):
    image_path = PROJECT_ROOT / "data" / "raw" / lake_name / image_name
    mask_path = PROJECT_ROOT / "data" / "masks" / "water" / lake_name / f"{Path(image_name).stem}_water_mask.tif"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Water mask not found: {mask_path}")

    model = model or load_model()

    with rasterio.open(str(image_path)) as src:
        bands = read_required_bands(src)
        image = np.stack([
            bands["B3"],
            bands["B4"],
            bands["B5"],
            bands["B8"],
            bands["B11"],
        ], axis=-1)
        # Training patches were stretched per image and band; inference has to
        # match that contract or every reflectance value above 1 saturates.
        image = normalize_bands(image)
        transform = src.transform
        crs = src.crs
        height, width = image.shape[:2]

    with rasterio.open(str(mask_path)) as mask_src:
        mask = mask_src.read(1).astype(np.uint8)

    if mask.shape[0] < height or mask.shape[1] < width:
        raise ValueError(
            f"Water mask {mask_path.name} ({mask.shape[1]}x{mask.shape[0]}) is smaller than "
            f"{image_name} ({width}x{height})"
        )
    mask = mask[:height, :width]

    quality_map = np.zeros((height, width), dtype=np.uint8)
    rows = []
    skipped = 0
    for row, col in extract_patch_candidates(image, mask):
        patch = image[row: row + PATCH_SIZE, col: col + PATCH_SIZE, :]
        patch_mask = mask[row: row + PATCH_SIZE, col: col + PATCH_SIZE]
        if patch.shape[0] == 0 or patch.shape[1] == 0:
            continue
        if not np.any(patch_mask):
            continue
        try:
            label, confidence, pred_idx = predict_single_patch(model, patch)
        except Exception as exc:
            skipped += 1
            print(f"Skipping patch ({row}, {col}) of {lake_name}/{image_name}: {exc}", file=sys.stderr)
            continue

        quality_map[row: row + PATCH_SIZE, col: col + PATCH_SIZE] = np.where(
            patch_mask > 0,
            pred_idx + 1,
            0,
        )

        rows.append({
            "lake": lake_name,
            "image": image_name,
            "row": row,
            "col": col,
            "prediction": label,
            "confidence": round(confidence, 6),
            "severity": SEVERITY_MAP[label],
        })

    if skipped:
        print(
            f"Warning: {skipped} patch(es) of {lake_name}/{image_name} could not be classified and were left unclassified.",
            file=sys.stderr,
        )

    output_dir = PROJECT_ROOT / "outputs" / "quality_predictions" / lake_name
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{Path(image_name).stem}_quality_predictions.csv"
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(prediction_path, index=False)

    gis_dir = PROJECT_ROOT / "outputs" / "gis" / "quality_maps"
    gis_dir.mkdir(parents=True, exist_ok=True)
    quality_path = gis_dir / f"{Path(image_name).stem}_quality.tif"
    with rasterio.open(
        str(quality_path),
        "w",
        driver="GTiff",
        height=quality_map.shape[0],
        width=quality_map.shape[1],
        count=1,
        dtype=quality_map.dtype,
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(quality_map, 1)

    return pd.DataFrame(rows, columns=CSV_COLUMNS), quality_map


def process_lake(lake_name: str, model=None, image_names=None):
    if image_names is None:
        lake_dir = PROJECT_ROOT / "data" / "raw" / lake_name
        image_names = sort_acquisitions(p.name for p in lake_dir.glob("*.tif"))

    model = model or load_model()
    summaries = []
    for image_name in image_names:
        pred_df, _ = process_image(lake_name, image_name, model=model)
        summaries.append(pred_df)
    return summaries


def process_all(model=None):
    model = model or load_model()
    lakes = sorted((PROJECT_ROOT / "data" / "raw").iterdir())
    all_summaries = []
    for lake_dir in lakes:
        if lake_dir.is_dir():
            lake_name = lake_dir.name
            all_summaries.extend(process_lake(lake_name, model=model))
    return all_summaries


def default_image_names():
    raw_root = PROJECT_ROOT / "data" / "raw"
    with open(PROJECT_ROOT / "data" / "datasets" / "efficientnet_dataset" / "image_split.json", "r", encoding="utf-8") as f:
        split_map = json.load(f)
    names = []
    for lake_name, temps in split_map.items():
        for split_name, file_list in temps.items():
            names.extend((lake_name, img) for img in sort_acquisitions(file_list))
    return names
