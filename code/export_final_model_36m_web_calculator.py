#!/usr/bin/env python3
"""Export the 36-month Model 2 + Logistic Regression artifact for the web calculator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBAPP_DIR = PROJECT_ROOT / "webapp"
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

from model_runtime import ARTIFACT_PATH, METADATA_PATH, export_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Refit and overwrite any existing artifact.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = export_artifact(force=args.force)
    payload = {
        "artifact_path": str(ARTIFACT_PATH),
        "metadata_path": str(METADATA_PATH),
        "model_name": artifact["metadata"]["model_name"],
        "horizon_months": artifact["metadata"]["horizon_months"],
        "n_features": len(artifact["metadata"]["features"]),
        "n_rows": artifact["metadata"]["training_summary"]["n_rows"],
        "n_events": artifact["metadata"]["training_summary"]["n_events"],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
