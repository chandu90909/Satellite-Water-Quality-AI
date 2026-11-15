# Satellite Water Quality Proxy Monitoring Using Deep Learning

This project estimates satellite-derived water-quality proxy classes from Sentinel-2 imagery using a two-stage deep-learning workflow:

1. A U-Net water-mask model isolates water pixels from land and background.
2. An EfficientNet-B0 classifier assigns each water patch to one of three proxy classes:
   - Good
   - Moderate
   - Poor

The current workflow preserves the existing trained checkpoints and adds the missing inference, severity, temporal, and validation layers around them.

## Repository structure
- `data/raw`: Sentinel-2 TIFF inputs by lake.
- `data/masks/water`: generated water masks.
- `data/datasets`: prepared patch and sampling datasets.
- `models/unet`: U-Net checkpoint.
- `models/efficientnet`: EfficientNet checkpoint and evaluation artifacts.
- `classification/inference`: classification entrypoints and inference logic.
- `segmentation/inference`: U-Net mask validation and inference entrypoint.
- `severity`: severity aggregation and summary logic.
- `temporal`: temporal trend plotting and change detection.
- `outputs`: prediction CSVs, GIS maps, severity summaries, and temporal charts.

## Quick start from GitHub

The code and trained checkpoints are stored in the repository. Large Sentinel-2 inputs, prepared datasets, and generated outputs are intentionally excluded, so a fresh clone can run the dashboard immediately and regenerate analysis products with the pipeline.

```bash
git lfs install
git clone https://github.com/chandu90909/Satellite-Water-Quality-AI.git
cd Satellite-Water-Quality-AI
py -3.11 -m venv .venv311
.venv311\Scripts\python.exe -m pip install -r requirements.txt
.venv311\Scripts\python.exe backend\server.py
```

Open `http://127.0.0.1:8000`. The two model checkpoints are downloaded through Git LFS. To populate imagery and dashboard results, configure Earth Engine as described below and run a lake through `pipeline/run_lake.py`, or point the pipeline at imagery already on disk.

## Inference commands

Run a single lake image:

```bash
python classification/inference/run_quality_inference.py --lake Hussain_Sagar --image Hussain_Sagar_1.tif
```

Run all images:

```bash
python classification/inference/run_quality_inference.py --all
```

Validate U-Net water-mask outputs:

```bash
python segmentation/inference/run_unet_inference.py
```

Generate severity summaries:

```bash
python severity/severity_mapper.py --input outputs/quality_predictions --output outputs/severity
```

Generate temporal summaries and charts for all lakes:

```bash
python temporal/temporal_analysis.py
```

Validate prediction tables and geospatial output consistency:

```bash
python reports/validate_outputs.py
```

Generate class-specific Grad-CAM overlays from the existing checkpoint and prediction CSVs:

```bash
python explainability/gradcam/generate_gradcam.py --all
```

Start the local dashboard and API:

```bash
python backend/server.py
```

Open `http://127.0.0.1:8000` in a browser. Select a lake and acquisition to inspect quality proportions, spatial maps, temporal trends, and available Grad-CAM explanations.

Master project runner:

```bash
python run_project.py --stage classification
python run_project.py --stage severity
python run_project.py --stage gradcam
python run_project.py --stage temporal
python run_project.py --stage validation
python run_project.py --all
```

The `--all` run executes classification, severity mapping, Grad-CAM generation, temporal analysis, and output validation in sequence. Classification and Grad-CAM use the existing EfficientNet checkpoint and do not retrain the model. Start the dashboard separately with `python backend/server.py`.

## Analyse a new lake (Sentinel-2 via Earth Engine)

Any lake and time period can be analysed end to end. Imagery is downloaded from Google Earth Engine and then processed by the same water-mask, classification, severity, Grad-CAM, temporal and validation stages as the bundled lakes.

One-time Earth Engine setup:

1. Create and register a Earth Engine Cloud project. The registration page can create, enable and register the project in one flow:

   - Open <https://code.earthengine.google.com/register> and sign in with the same Google account you authenticate with.
   - Choose **Create a new Cloud project**, select **Noncommercial** use (research/teaching), accept the terms and register.
   - The page enables the Earth Engine API and shows the new project ID (format `words-123456`).
   - Manual route: create a project in the [Cloud Console](https://console.cloud.google.com/projectcreate), run
     `gcloud services enable earthengine.googleapis.com --project PROJECT_ID`, then register it at
     `https://code.earthengine.google.com/register?project=PROJECT_ID`.
   - Since April 2026 noncommercial quotas require an active billing account. Earth Engine usage is not charged on a
     noncommercial project, but other Cloud services in that project can be.
2. Authenticate the local client:

   ```bash
   .venv311\Scripts\earthengine.exe authenticate
   ```

   or create a service account key and set `EE_SERVICE_ACCOUNT` and `EE_PRIVATE_KEY_FILE`.

   If the stored `earthengine` login belongs to a Google account that cannot use
   the project, the client automatically falls back to the account signed in
   with `gcloud auth login`, fetching a fresh token for each run. Run
   `earthengine authenticate --force` to make the standalone login match the
   project owner instead.
3. Expose the project ID, in order of precedence:

   - `--project` on the command line,
   - the `EE_PROJECT` environment variable,
   - `config/earth_engine.json` with `{"project": "your-project-id"}` (used by the CLI and the dashboard).

Verify access before downloading:

```bash
python gee/download_lake.py --check
```

Run the full workflow for a lake and period:

```bash
python pipeline/run_lake.py --lake "Kolleru Lake" --start 2025-01-01 --end 2025-03-31
```

Reuse imagery that is already on disk (no Earth Engine call):

```bash
python pipeline/run_lake.py --lake Kolleru_Lake --skip-download
```

Download imagery only, with optional scene-level cadence, cloud limit and boundary file:

```bash
python gee/download_lake.py --lake "Kolleru Lake" --start 2025-01-01 --end 2025-03-31 \
  --cadence monthly --max-cloud 30 --boundary data/boundaries/Kolleru_Lake.geojson
```

What the download stage does:

- Resolves the lake boundary from an explicit GeoJSON file, the cached boundary in `data/boundaries`, the extent of imagery already downloaded for that lake, then OpenStreetMap Nominatim (Photon as fallback).
- Queries `COPERNICUS/S2_SR_HARMONIZED`, filters by date, boundary and `CLOUDY_PIXEL_PERCENTAGE`, and masks clouds, shadows, cirrus, snow and defective pixels with the SCL band.
- Exports B3, B4, B5, B8, B11 and SCL at 10 m in EPSG:4326 to `data/raw/<Lake>/<Lake>_<YYYYMM>.tif` (monthly median composite) or `<Lake>_<YYYYMMDD>.tif` (every clear acquisition), plus `acquisition_manifest.json`.

Dashboard: enter a lake name and period in **Satellite Water Quality Monitoring**, choose compositing and cloud cover, and press **Run Analysis**. The job streams progress and log lines, then refreshes the same maps, metrics, temporal chart, Grad-CAM views and acquisition table. Analyses run one at a time.

## Scientific framing
This project produces proxy-derived quality classifications from Sentinel-2 spectral information rather than laboratory-measured water chemistry. Terms such as "proxy-derived water-quality condition" and "satellite-derived water-quality proxy classification" are used throughout, consistent with the project methodology.

## Existing trained models
- `models/unet/unet_best.pth`
- `models/efficientnet/efficientnet_b0_best.pth`

## Environment
- Use Python 3.11-3.13 for the model stages: torch 2.8 has no Python 3.14 wheels.
- `pip install -r requirements.txt` then `python -m pytest tests` to run the checks.
- Every stage derives its paths from the repository location, so the project can run from any drive.

## Notes
- No retraining was performed as part of the remaining workflow.
- Existing data leakage safeguards and image-level splits were preserved.
- GeoTIFF outputs keep the original CRS and transform whenever available.
- Inference stretches every scene per band with the same 2/98 percentile transform used to build the training patches. The earlier clip-only preprocessing saturated reflectance values above 1.0 and collapsed almost every prediction to "Good".
- New lakes are processed with the same 128 px tile grid, 5-band ordering and NDWI/MNDWI water-mask thresholds as the bundled lakes, so their outputs are directly comparable.
