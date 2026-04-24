#!/usr/bin/env python3
"""Runtime utilities for the 36-month ADNI Model 2 + Logistic Regression calculator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "mci_ad" / "mci_ad_dataset_all.csv"
MRI_FEATURE_PATH = PROJECT_ROOT / "results" / "mri_feature_selection" / "data" / "mri_final_candidates_main.csv"
MRI_POOL_PATH = PROJECT_ROOT / "results" / "mri_feature_selection" / "data" / "mri_candidate_pool.csv"
OOF_PREDICTION_PATH = (
    PROJECT_ROOT
    / "results"
    / "clinical_model_comparison"
    / "horizon_36m"
    / "model2"
    / "data"
    / "logistic_regression_oof_predictions.csv"
)
ARTIFACT_ROOT = PROJECT_ROOT / "results" / "web_calculator_36m_model2_lr_v1" / "artifacts"
ARTIFACT_PATH = ARTIFACT_ROOT / "final_model_36m_model2_lr.joblib"
METADATA_PATH = ARTIFACT_ROOT / "final_model_36m_model2_lr_metadata.json"

SEED = 42
HORIZON_MONTHS = 36
LABEL_COLUMN = f"label_{HORIZON_MONTHS}m"
CATEGORICAL_FEATURES = ["sex", "demog_ptethcat", "demog_ptraccat"]
MODEL1_BASE_FEATURES = [
    "age_at_baseline",
    "sex",
    "demog_pteducat",
    "demog_ptethcat",
    "demog_ptraccat",
    "mmse_mmscore",
    "adas_total13",
    "faq_faqtotal",
]

FEATURE_LABELS = {
    "age_at_baseline": "Age at baseline",
    "sex": "Sex",
    "demog_pteducat": "Education (years)",
    "demog_ptethcat": "Ethnicity code",
    "demog_ptraccat": "Race code",
    "mmse_mmscore": "MMSE",
    "adas_total13": "ADAS-Cog13",
    "faq_faqtotal": "FAQ total",
    "ST130TA": "ST130TA: right middle temporal cortical thickness",
    "ST88SV": "ST88SV: right hippocampal volume",
    "ST115CV": "ST115CV: right superior frontal gray-matter volume",
    "ST99TA": "ST99TA: right insular cortical thickness",
    "ST73CV": "ST73CV: right caudal anterior cingulate gray-matter volume",
}

FEATURE_HELP = {
    "age_at_baseline": "Baseline age in years.",
    "sex": "Use the same coding style as the ADNI training data.",
    "demog_pteducat": "Completed years of education.",
    "demog_ptethcat": "ADNI ethnicity code used during model training.",
    "demog_ptraccat": "ADNI race code used during model training.",
    "mmse_mmscore": "Mini-Mental State Examination total score.",
    "adas_total13": "ADAS-Cog13 total score.",
    "faq_faqtotal": "Functional Activities Questionnaire total score.",
    "ST130TA": "MRI-derived regional cortical thickness measure.",
    "ST88SV": "MRI-derived regional hippocampal/subcortical volume measure.",
    "ST115CV": "MRI-derived regional gray-matter volume measure.",
    "ST99TA": "MRI-derived regional cortical thickness measure.",
    "ST73CV": "MRI-derived regional gray-matter volume measure.",
}

BASE_WIDGET_CONFIG = {
    "age_at_baseline": {"min": 50.0, "max": 95.0, "step": 1.0},
    "demog_pteducat": {"min": 0.0, "max": 30.0, "step": 1.0},
    "mmse_mmscore": {"min": 0.0, "max": 30.0, "step": 1.0},
    "adas_total13": {"min": 0.0, "max": 70.0, "step": 0.1},
    "faq_faqtotal": {"min": 0.0, "max": 30.0, "step": 1.0},
}

MODEL_CARD = {
    "display_name": "36-month MCI-to-AD conversion calculator",
    "model_name": "ADNI final Model 2 + Logistic Regression",
    "internal_validation": {
        "cohort": "ADNI internal out-of-fold validation",
        "n": 893,
        "events": 277,
        "auc": 0.859,
        "auc_ci": [0.833, 0.883],
        "brier": 0.139,
        "brier_ci": [0.124, 0.153],
    },
    "note": (
        "This calculator follows the research-model definition used for the ADNI internal final model. "
        "The locked NACC external analysis used a different common-clinical model because ADAS-Cog13 was unavailable."
    ),
}


def _read_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    frame.columns = [column.strip() for column in frame.columns]
    return frame


def _load_selected_mri_features() -> list[str]:
    frame = pd.read_csv(MRI_FEATURE_PATH)
    features = [str(value).strip() for value in frame["feature"].tolist() if str(value).strip()]
    if not features:
        raise ValueError(f"No MRI features found in {MRI_FEATURE_PATH}")
    return features


def build_model2_features() -> list[str]:
    return [*MODEL1_BASE_FEATURES, *_load_selected_mri_features()]


def _prepare_dataset(features: list[str]) -> pd.DataFrame:
    frame = _read_table(DATASET_PATH)
    mri_pool = _read_table(MRI_POOL_PATH)
    selected_mri_cols = sorted([feature for feature in features if feature.startswith("ST")])
    join_columns = [column for column in ["RID", *selected_mri_cols] if column in mri_pool.columns]
    if "RID" not in join_columns:
        raise ValueError(f"MRI pool is missing RID: {MRI_POOL_PATH}")
    mri_pool = mri_pool[join_columns].drop_duplicates(subset=["RID"], keep="first")
    frame = frame.merge(mri_pool, on="RID", how="left")

    required_columns = [
        "RID",
        "baseline_date",
        "time_to_ad_months",
        "last_followup_months",
        LABEL_COLUMN,
        "strict_label_36m",
        *features,
    ]
    available_columns = [column for column in required_columns if column in frame.columns]
    data = frame[available_columns].copy()
    for column in data.columns:
        if column not in {"RID", *CATEGORICAL_FEATURES}:
            try:
                data[column] = pd.to_numeric(data[column], errors="raise")
            except Exception:
                pass

    data = data[data[LABEL_COLUMN].notna()].copy()
    data["label"] = pd.to_numeric(data[LABEL_COLUMN], errors="coerce").astype(int)
    data["event"] = np.where(pd.to_numeric(data["time_to_ad_months"], errors="coerce").notna(), 1, 0)
    data["time_months"] = np.where(
        data["event"] == 1,
        pd.to_numeric(data["time_to_ad_months"], errors="coerce"),
        pd.to_numeric(data["last_followup_months"], errors="coerce"),
    )
    data = data[data["time_months"].notna()].copy()
    return data


def build_preprocessor(features: list[str]) -> ColumnTransformer:
    categorical_columns = [column for column in features if column in CATEGORICAL_FEATURES]
    numeric_columns = [column for column in features if column not in categorical_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def _make_pipeline(features: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("model", LogisticRegression(max_iter=5000, solver="lbfgs", random_state=SEED)),
        ]
    )


def _safe_mode(series: pd.Series, fallback: str) -> str:
    cleaned = series.dropna().astype(str)
    cleaned = cleaned[~cleaned.isin(["", "-4", "-4.0", "nan", "None"])]
    if cleaned.empty:
        return fallback
    return str(cleaned.mode().iloc[0])


def _numeric_widget_limits(feature: str, series: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if feature in BASE_WIDGET_CONFIG:
        return dict(BASE_WIDGET_CONFIG[feature])
    lower = float(np.nanpercentile(numeric, 1)) if len(numeric) else 0.0
    upper = float(np.nanpercentile(numeric, 99)) if len(numeric) else 1.0
    if np.isclose(lower, upper):
        lower = float(np.nanmin(numeric)) if len(numeric) else 0.0
        upper = float(np.nanmax(numeric)) if len(numeric) else 1.0
    return {
        "min": float(np.floor(lower * 1000.0) / 1000.0),
        "max": float(np.ceil(upper * 1000.0) / 1000.0),
        "step": 0.001,
    }


def _collect_feature_metadata(data: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    widget_config: dict[str, dict[str, Any]] = {}
    category_options: dict[str, list[str]] = {}

    for feature in features:
        series = data[feature]
        if feature in CATEGORICAL_FEATURES:
            cleaned = series.dropna().astype(str)
            cleaned = cleaned[~cleaned.isin(["", "-4", "-4.0", "nan", "None"])]
            options = sorted(cleaned.unique().tolist())
            defaults[feature] = _safe_mode(series, fallback=options[0] if options else "")
            category_options[feature] = options
            widget_config[feature] = {"options": options}
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            defaults[feature] = float(np.nanmedian(numeric))
            widget_config[feature] = _numeric_widget_limits(feature, numeric)

    return {
        "defaults": defaults,
        "widget_config": widget_config,
        "category_options": category_options,
    }


def _aggregate_contributions(
    transformed_row: np.ndarray,
    transformed_names: list[str],
    coefficients: np.ndarray,
    background_mean: np.ndarray,
    features: list[str],
) -> list[dict[str, Any]]:
    raw_contributions = (transformed_row - background_mean) * coefficients.reshape(1, -1)
    rows: list[dict[str, Any]] = []

    for feature in features:
        numeric_name = f"num__{feature}"
        if numeric_name in transformed_names:
            indices = [transformed_names.index(numeric_name)]
        else:
            prefix = f"cat__{feature}_"
            indices = [index for index, name in enumerate(transformed_names) if name.startswith(prefix)]
        if not indices:
            continue
        contribution = float(raw_contributions[:, indices].sum())
        rows.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "contribution_log_odds": contribution,
            }
        )

    rows.sort(key=lambda row: abs(row["contribution_log_odds"]), reverse=True)
    return rows


def fit_model_artifact() -> dict[str, Any]:
    features = build_model2_features()
    data = _prepare_dataset(features)
    x = data[features].copy()
    y = data["label"].to_numpy(dtype=int)

    pipeline = _make_pipeline(features)
    pipeline.fit(x, y)

    oof_predictions = pd.read_csv(OOF_PREDICTION_PATH)
    lower_threshold, upper_threshold = oof_predictions["oof_probability"].quantile([1.0 / 3.0, 2.0 / 3.0]).tolist()

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed_x = preprocessor.transform(x)
    transformed_names = list(preprocessor.get_feature_names_out())
    coefficients = np.asarray(model.coef_).ravel()
    background_mean = np.asarray(transformed_x).mean(axis=0)
    base_log_odds = float(model.intercept_[0] + np.dot(background_mean, coefficients))

    feature_metadata = _collect_feature_metadata(x, features)

    metadata = {
        "display_name": MODEL_CARD["display_name"],
        "model_name": MODEL_CARD["model_name"],
        "horizon_months": HORIZON_MONTHS,
        "features": features,
        "feature_labels": FEATURE_LABELS,
        "feature_help": FEATURE_HELP,
        "defaults": feature_metadata["defaults"],
        "widget_config": feature_metadata["widget_config"],
        "category_options": feature_metadata["category_options"],
        "risk_group_thresholds": {
            "low_upper": float(lower_threshold),
            "high_lower": float(upper_threshold),
        },
        "performance": MODEL_CARD["internal_validation"],
        "model_note": MODEL_CARD["note"],
        "training_summary": {
            "n_rows": int(len(x)),
            "n_events": int(y.sum()),
            "event_rate": float(y.mean()),
        },
        "transformed_feature_names": transformed_names,
        "background_mean": background_mean.tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": float(model.intercept_[0]),
        "base_log_odds": base_log_odds,
        "source_paths": {
            "dataset": str(DATASET_PATH),
            "mri_feature_file": str(MRI_FEATURE_PATH),
            "mri_pool": str(MRI_POOL_PATH),
            "oof_predictions": str(OOF_PREDICTION_PATH),
        },
    }

    return {"pipeline": pipeline, "metadata": metadata}


def export_artifact(force: bool = False) -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    if ARTIFACT_PATH.exists() and not force:
        return load_artifact()

    artifact = fit_model_artifact()
    joblib.dump(artifact, ARTIFACT_PATH)
    METADATA_PATH.write_text(json.dumps(artifact["metadata"], indent=2), encoding="utf-8")
    return artifact


def load_artifact() -> dict[str, Any]:
    if not ARTIFACT_PATH.exists():
        return export_artifact(force=True)
    return joblib.load(ARTIFACT_PATH)


def build_input_frame(inputs: dict[str, Any], metadata: dict[str, Any]) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for feature in metadata["features"]:
        value = inputs.get(feature, metadata["defaults"][feature])
        if feature in CATEGORICAL_FEATURES:
            row[feature] = str(value)
        else:
            row[feature] = float(value)
    return pd.DataFrame([row], columns=metadata["features"])


def assign_risk_group(probability: float, metadata: dict[str, Any]) -> str:
    thresholds = metadata["risk_group_thresholds"]
    if probability <= thresholds["low_upper"]:
        return "low"
    if probability >= thresholds["high_lower"]:
        return "high"
    return "intermediate"


def score_inputs(inputs: dict[str, Any], artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    runtime = load_artifact() if artifact is None else artifact
    metadata = runtime["metadata"]
    pipeline: Pipeline = runtime["pipeline"]
    row = build_input_frame(inputs, metadata)

    transformed_row = pipeline.named_steps["preprocessor"].transform(row)
    transformed_names = metadata["transformed_feature_names"]
    coefficients = np.asarray(metadata["coefficients"], dtype=float)
    background_mean = np.asarray(metadata["background_mean"], dtype=float)
    intercept = float(metadata["intercept"])

    log_odds = float(intercept + np.dot(np.asarray(transformed_row).ravel(), coefficients))
    probability = float(1.0 / (1.0 + np.exp(-log_odds)))
    risk_group = assign_risk_group(probability, metadata)
    contributions = _aggregate_contributions(
        transformed_row=np.asarray(transformed_row),
        transformed_names=transformed_names,
        coefficients=coefficients,
        background_mean=background_mean,
        features=metadata["features"],
    )

    return {
        "probability": probability,
        "risk_group": risk_group,
        "log_odds": log_odds,
        "base_log_odds": float(metadata["base_log_odds"]),
        "input_row": row.to_dict(orient="records")[0],
        "contributions": contributions,
    }


def example_inputs() -> dict[str, Any]:
    artifact = load_artifact()
    return dict(artifact["metadata"]["defaults"])
