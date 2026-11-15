# Project Status

This repository keeps the working model checkpoints and dataset artifacts intact while adding the missing inference, severity, temporal, and validation stages around them.

## Production checkpoints preserved
- `models/unet/unet_best.pth`
- `models/efficientnet/efficientnet_b0_best.pth`

## Key rules followed
- No retraining of the U-Net model unless explicitly requested.
- No retraining of EfficientNet unless explicitly requested.
- No overwrite of existing project datasets or labels.
- The 5-band architecture is preserved: B3, B4, B5, B8, B11.
- The 128x128 patch pipeline remains consistent with the trained EfficientNet checkpoint.
- Proxy-derived outputs are treated as satellite-derived quality proxies, not laboratory measurements.

## Implemented remaining workflow
- Classification inference entrypoint at `classification/inference/run_quality_inference.py`
- Severity aggregation at `severity/severity_mapper.py`
- Master project runner at `run_project.py`
- U-Net inference/status check at `segmentation/inference/run_unet_inference.py`
- Temporal summary plotting at `temporal/temporal_analysis.py`

## Verified on the current environment
- Single-image EfficientNet inference on `Hussain_Sagar/Hussain_Sagar_1.tif` succeeded.
- The corresponding prediction CSV and GeoTIFF output were written under the project output directories.
- The severity summary script generated the expected summary CSV successfully.
