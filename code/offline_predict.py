#!/usr/bin/env python3
"""Offline batch inference for the 36-month ADNI risk calculator.

This script loads the exported joblib artifact directly, so it can be used
without the original training data or the Streamlit UI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "final_model_36m_model2_lr.joblib"


def _load_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact or "metadata" not in artifact:
        raise ValueError(f"Unexpected artifact structure in {path}")
    return artifact


def _load_feature_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Feature map must be a JSON object of model_feature -> source_column.")
    return {str(key): str(value) for key, value in payload.items()}


def _prepare_input_frame(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    feature_map: dict[str, str],
    strict: bool,
) -> tuple[pd.DataFrame, list[str]]:
    defaults = metadata["defaults"]
    categorical_features = set(metadata.get("category_options", {}).keys())
    prepared = pd.DataFrame(index=frame.index)
    missing_features: list[str] = []

    for feature in metadata["features"]:
        source_column = feature_map.get(feature, feature)
        if source_column in frame.columns:
            series = frame[source_column]
        elif feature in frame.columns:
            series = frame[feature]
        else:
            if strict:
                missing_features.append(feature)
                continue
            series = pd.Series([defaults[feature]] * len(frame), index=frame.index)
            missing_features.append(feature)

        if feature in categorical_features:
            filled = series.copy()
            filled = filled.where(~filled.isna(), defaults[feature])
            prepared[feature] = filled.astype(str)
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            prepared[feature] = numeric.fillna(float(defaults[feature])).astype(float)

    if strict and missing_features:
        raise KeyError(
            "Missing required model features: " + ", ".join(sorted(set(missing_features)))
        )

    return prepared, sorted(set(missing_features))


def score_frame(
    frame: pd.DataFrame,
    runtime: dict[str, Any],
    feature_map: dict[str, str] | None = None,
    strict: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    metadata = runtime["metadata"]
    pipeline = runtime["pipeline"]
    prepared, missing_features = _prepare_input_frame(
        frame=frame,
        metadata=metadata,
        feature_map=feature_map or {},
        strict=strict,
    )

    probabilities = pipeline.predict_proba(prepared)[:, 1]
    log_odds = pipeline.decision_function(prepared)
    thresholds = metadata["risk_group_thresholds"]

    risk_group = np.where(
        probabilities <= float(thresholds["low_upper"]),
        "low",
        np.where(probabilities >= float(thresholds["high_lower"]), "high", "intermediate"),
    )

    scored = frame.copy()
    scored["probability"] = probabilities
    scored["probability_pct"] = probabilities * 100.0
    scored["log_odds"] = log_odds
    scored["risk_group"] = risk_group
    return scored, missing_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline batch prediction for the ADNI 36-month MCI-to-AD calculator."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input CSV file to score.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV file. Defaults to <input_stem>_scored.csv next to the input file.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
        help="Path to final_model_36m_model2_lr.joblib.",
    )
    parser.add_argument(
        "--feature-map",
        type=Path,
        default=None,
        help=(
            "Optional JSON object mapping model feature names to source column names "
            "in the input file."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any required model feature is missing from the input file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = _load_artifact(args.artifact)
    feature_map = _load_feature_map(args.feature_map)
    input_frame = pd.read_csv(args.input)

    scored, missing_features = score_frame(
        frame=input_frame,
        runtime=artifact,
        feature_map=feature_map,
        strict=args.strict,
    )

    output_path = args.output
    if output_path is None:
        output_path = args.input.with_name(f"{args.input.stem}_scored.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)

    metadata = artifact["metadata"]
    summary = {
        "artifact": str(args.artifact),
        "input": str(args.input),
        "output": str(output_path),
        "rows": int(len(scored)),
        "model_name": metadata["model_name"],
        "horizon_months": metadata["horizon_months"],
        "risk_group_thresholds": metadata["risk_group_thresholds"],
        "missing_features_filled_with_defaults": missing_features,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
