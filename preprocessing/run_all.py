import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


SCRIPTS = [
    "mndwi.py",
    "ndwi.py",
    "ndvi.py",
    "ndci.py",
    "turbidity.py",
    "chlorophyll.py",
]

raw_root = PROJECT_ROOT / "data" / "raw"
if not raw_root.exists() or not any(raw_root.iterdir()):
    print(f"No source imagery found under {raw_root}.")
    sys.exit(1)

for script in SCRIPTS:

    print("\n" + "=" * 70)
    print(f"RUNNING: {script}")
    print("=" * 70)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            f"preprocessing.indices.{script[:-3]}"
        ],
        cwd=str(PROJECT_ROOT)
    )

    if result.returncode != 0:
        print(f"\nERROR: {script} failed.")
        sys.exit(result.returncode)


print("\n" + "=" * 70)
print("REMAINING PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)
