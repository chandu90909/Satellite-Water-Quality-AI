import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.ordering import acquisition_sort_key, sort_acquisitions
from common.patch_utils import pad_to_size
from severity.severity_mapper import summarize_predictions
from temporal import temporal_analysis


def test_acquisition_sort_key_orders_numerically():
    names = ["Hussain_Sagar_10.tif", "Hussain_Sagar_2.tif", "Hussain_Sagar_1.tif"]
    assert sort_acquisitions(names) == [
        "Hussain_Sagar_1.tif",
        "Hussain_Sagar_2.tif",
        "Hussain_Sagar_10.tif",
    ]
    assert acquisition_sort_key("Hussain_Sagar_7") < acquisition_sort_key("Hussain_Sagar_12")


def test_temporal_summary_is_chronological(tmp_path, monkeypatch):
    lake_dir = tmp_path / "Hussain_Sagar"
    lake_dir.mkdir()
    for number in (1, 2, 10, 11):
        pd.DataFrame({"prediction": ["Good", "Poor"]}).to_csv(
            lake_dir / f"Hussain_Sagar_{number}_quality_predictions.csv", index=False
        )
    monkeypatch.setattr(temporal_analysis, "PREDICTION_ROOT", tmp_path)

    summary = temporal_analysis.summarize_lake_quality("Hussain_Sagar")

    assert list(summary["image"]) == [
        "Hussain_Sagar_1",
        "Hussain_Sagar_2",
        "Hussain_Sagar_10",
        "Hussain_Sagar_11",
    ]


def test_severity_summary_reports_percentages_and_patch_counts(tmp_path):
    csv_path = tmp_path / "Hussain_Sagar_1_quality_predictions.csv"
    pd.DataFrame({
        "lake": ["Hussain_Sagar"] * 4,
        "image": ["Hussain_Sagar_1.tif"] * 4,
        "row": [0, 0, 128, 128],
        "col": [0, 128, 0, 128],
        "prediction": ["Good", "Good", "Good", "Poor"],
        "confidence": [0.9, 0.8, 0.7, 0.6],
        "severity": ["Low", "Low", "Low", "High"],
    }).to_csv(csv_path, index=False)

    summary = summarize_predictions(csv_path)

    assert summary["classified_patches"] == 4
    assert summary["good_percentage"] == pytest.approx(75.0)
    assert summary["poor_percentage"] == pytest.approx(25.0)
    assert summary["dominant_quality"] == "Good"
    assert summary["dominant_severity"] == "Low"


def test_pad_to_size_pads_edge_tiles():
    padded = pad_to_size(np.zeros((90, 128, 5), dtype=np.float32), 128)

    assert padded.shape == (128, 128, 5)
    assert not padded[90:].any()


def test_pipeline_sources_have_no_hardcoded_drive_paths():
    root = Path(__file__).resolve().parents[1]
    ignored = {"python_packages", "data", "outputs", "__pycache__", ".pytest_cache"}
    offenders = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if any(part in ignored or part.startswith(".venv") for part in relative.parts):
            continue
        if re.search(r"[A-Za-z]:\\\\Satellite", path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(relative.as_posix())

    assert offenders == []
