#!/usr/bin/env python3
"""Minimal Streamlit page for offline deployment of the 36-month calculator."""

from __future__ import annotations

import json
from pathlib import Path
import html
import sys
from typing import Any
from textwrap import dedent

import joblib
import pandas as pd
import streamlit as st


WEBAPP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEBAPP_DIR.parent
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

from model_runtime import CATEGORICAL_FEATURES, FEATURE_LABELS, score_inputs


ARTIFACT_CANDIDATES = [
    PROJECT_ROOT / "artifacts" / "final_model_36m_model2_lr.joblib",
    PROJECT_ROOT / "results" / "web_calculator_36m_model2_lr_v1" / "artifacts" / "final_model_36m_model2_lr.joblib",
]

FEATURE_GROUPS = [
    ("Demographics", ["age_at_baseline", "sex", "demog_pteducat", "demog_ptethcat", "demog_ptraccat"]),
    ("Cognitive", ["mmse_mmscore", "adas_total13", "faq_faqtotal"]),
    ("MRI", ["ST130TA", "ST88SV", "ST115CV", "ST99TA", "ST73CV"]),
]

LANG_OPTIONS = {"en": "English", "zh": "\u4e2d\u6587"}

TEXT = {
    "en": {
        "hero_title": "ADNI 36-Month Minimal Runner",
        "hero_subtitle": "A lightweight browser page for offline prediction. Load the local artifact and run single-case or CSV scoring.",
        "hero_eyebrow": "Locked research calculator",
        "hero_chip_offline": "Offline deployment",
        "hero_chip_artifact": "Local artifact only",
        "hero_chip_batch": "Single case + CSV",
        "model_label": "Model",
        "horizon_label": "Horizon",
        "ribbon_auc": "Internal AUC",
        "ribbon_brier": "Brier score",
        "ribbon_features": "Features",
        "ribbon_mode": "Mode",
        "ribbon_mode_value": "Research artifact",
        "local_artifact": "Local artifact: `artifacts/final_model_36m_model2_lr.joblib`. No training data or database connection is required for deployment.",
        "single_case": "Single case",
        "single_case_hint": "Fill the fields and click Predict.",
        "form_helper": "Structured around demographics, cognition, and MRI features.",
        "predict_button": "Predict",
        "how_it_works": "What this page does",
        "workflow_title": "Scoring flow",
        "feature_map_title": "Input landscape",
        "step_1": "Loads the exported `joblib` artifact from the local `artifacts/` folder.",
        "step_2": "Converts your inputs into the model feature schema.",
        "step_3": "Returns probability, risk group, and top feature contributions.",
        "minimal_note": "This is a deployment-minimal page. It intentionally avoids extra tabs, dashboard widgets, and any dependence on the original training dataset.",
        "prediction": "Prediction",
        "probability": "Probability",
        "risk_group": "Risk group",
        "log_odds": "Log-odds",
        "thresholds": "Risk thresholds from internal OOF tertiles: low <= {low:.3f}, high >= {high:.3f}.",
        "top_contributions": "Top feature contributions",
        "contrib_positive": "positive",
        "contrib_negative": "negative",
        "contrib_top": "Top",
        "contrib_items": "items",
        "contrib_note": "Relative contribution strength is scaled to the strongest feature in this case.",
        "entered_values": "Entered values",
        "feature": "Feature",
        "value": "Value",
        "batch_title": "Batch CSV scoring",
        "batch_desc": "Upload a CSV with the same feature columns to score multiple participants at once.",
        "download_template": "Download CSV template",
        "upload_csv": "Upload CSV",
        "score_csv": "Score CSV",
        "download_scored": "Download scored CSV",
        "missing_columns": "Missing columns were filled from training defaults: {cols}",
        "model_note": "Model note",
        "participant_id": "participant_id",
        "group_demographics": "Demographics",
        "group_cognitive": "Cognitive",
        "group_mri": "MRI",
        "language": "Language",
        "training_code": "training code",
        "sex_female": "Female",
        "sex_male": "Male",
        "select_hint": "Select the interface language.",
    },
    "zh": {
        "hero_title": "ADNI 36\u4e2a\u6708\u6700\u5c0f\u8fd0\u884c\u9875",
        "hero_subtitle": "\u4e00\u4e2a\u7528\u4e8e\u79bb\u7ebf\u9884\u6d4b\u7684\u8f7b\u91cf\u7f51\u9875\u3002\u52a0\u8f7d\u672c\u5730\u5de5\u4ef6\u540e\u5373\u53ef\u8fdb\u884c\u5355\u75c5\u4f8b\u6216 CSV \u6279\u91cf\u8bc4\u5206\u3002",
        "hero_eyebrow": "\u9501\u5b9a\u7248\u7814\u7a76\u8ba1\u7b97\u5668",
        "hero_chip_offline": "\u79bb\u7ebf\u90e8\u7f72",
        "hero_chip_artifact": "\u4ec5\u4f9d\u8d56\u672c\u5730\u5de5\u4ef6",
        "hero_chip_batch": "\u5355\u4f8b + CSV \u6279\u91cf",
        "model_label": "\u6a21\u578b",
        "horizon_label": "\u9884\u6d4b\u65f6\u9650",
        "ribbon_auc": "\u5185\u90e8AUC",
        "ribbon_brier": "Brier\u5206\u6570",
        "ribbon_features": "\u7279\u5f81\u6570",
        "ribbon_mode": "\u6a21\u5f0f",
        "ribbon_mode_value": "\u7814\u7a76\u5de5\u4ef6",
        "local_artifact": "\u672c\u5730\u5de5\u4ef6\uff1a`artifacts/final_model_36m_model2_lr.joblib`\u3002\u90e8\u7f72\u65f6\u4e0d\u9700\u8981\u8bad\u7ec3\u6570\u636e\u6216\u6570\u636e\u5e93\u8fde\u63a5\u3002",
        "single_case": "\u5355\u75c5\u4f8b",
        "single_case_hint": "\u586b\u5199\u5b57\u6bb5\u540e\u70b9\u51fb\u201c\u9884\u6d4b\u201d\u3002",
        "form_helper": "\u8f93\u5165\u533a\u57df\u6309\u4eba\u53e3\u5b66\u3001\u8ba4\u77e5\u548c MRI \u4e09\u7ec4\u7279\u5f81\u7ec4\u7ec7\u3002",
        "predict_button": "\u9884\u6d4b",
        "how_it_works": "\u9875\u9762\u529f\u80fd",
        "workflow_title": "\u8bc4\u5206\u6d41\u7a0b",
        "feature_map_title": "\u8f93\u5165\u6982\u89c8",
        "step_1": "\u4ece\u672c\u5730 `artifacts/` \u6587\u4ef6\u5939\u52a0\u8f7d\u5bfc\u51fa\u7684 `joblib` \u5de5\u4ef6\u3002",
        "step_2": "\u628a\u8f93\u5165\u503c\u8f6c\u6362\u6210\u6a21\u578b\u6240\u9700\u7684\u7279\u5f81\u683c\u5f0f\u3002",
        "step_3": "\u8f93\u51fa\u6982\u7387\u3001\u98ce\u9669\u5206\u7ec4\u548c\u4e3b\u8981\u7279\u5f81\u8d21\u732e\u3002",
        "minimal_note": "\u8fd9\u662f\u4e00\u4e2a\u6781\u7b80\u90e8\u7f72\u9875\uff0c\u523b\u610f\u4e0d\u4fdd\u7559\u989d\u5916\u6807\u7b7e\u9875\u3001\u4eea\u8868\u76d8\u7ec4\u4ef6\uff0c\u4e5f\u4e0d\u4f9d\u8d56\u539f\u59cb\u8bad\u7ec3\u6570\u636e\u8868\u3002",
        "prediction": "\u9884\u6d4b\u7ed3\u679c",
        "probability": "\u6982\u7387",
        "risk_group": "\u98ce\u9669\u5206\u7ec4",
        "log_odds": "\u5bf9\u6570\u51e0\u7387",
        "thresholds": "\u5185\u90e8 OOF \u4e09\u5206\u4f4d\u9608\u503c\uff1a\u4f4e\u98ce\u9669 <= {low:.3f}\uff0c\u9ad8\u98ce\u9669 >= {high:.3f}\u3002",
        "top_contributions": "\u4e3b\u8981\u7279\u5f81\u8d21\u732e",
        "contrib_positive": "\u6b63\u5411",
        "contrib_negative": "\u8d1f\u5411",
        "contrib_top": "\u6700\u5927\u5f71\u54cd",
        "contrib_items": "\u9879",
        "contrib_note": "\u76f8\u5bf9\u8d21\u732e\u5f3a\u5ea6\u5df2\u6309\u672c\u4f8b\u4e2d\u6700\u5927\u7279\u5f81\u8fdb\u884c\u7f29\u653e\u3002",
        "entered_values": "\u5df2\u8f93\u5165\u503c",
        "feature": "\u7279\u5f81",
        "value": "\u6570\u503c",
        "batch_title": "CSV \u6279\u91cf\u8bc4\u5206",
        "batch_desc": "\u4e0a\u4f20\u5305\u542b\u76f8\u540c\u7279\u5f81\u5217\u7684 CSV\uff0c\u53ef\u4e00\u6b21\u4e3a\u591a\u4e2a\u53c2\u4e0e\u8005\u6253\u5206\u3002",
        "download_template": "\u4e0b\u8f7d CSV \u6a21\u677f",
        "upload_csv": "\u4e0a\u4f20 CSV",
        "score_csv": "\u5f00\u59cb\u8bc4\u5206",
        "download_scored": "\u4e0b\u8f7d\u8bc4\u5206\u7ed3\u679c",
        "missing_columns": "\u7f3a\u5931\u5217\u5df2\u4f7f\u7528\u8bad\u7ec3\u9ed8\u8ba4\u503c\u8865\u9f50\uff1a{cols}",
        "model_note": "\u6a21\u578b\u8bf4\u660e",
        "participant_id": "\u53c2\u4e0e\u8005ID",
        "group_demographics": "\u4eba\u53e3\u5b66\u4fe1\u606f",
        "group_cognitive": "\u8ba4\u77e5\u91cf\u8868",
        "group_mri": "MRI \u4fe1\u606f",
        "language": "\u8bed\u8a00",
        "training_code": "\u8bad\u7ec3\u7f16\u7801",
        "sex_female": "\u5973",
        "sex_male": "\u7537",
        "select_hint": "\u8bf7\u9009\u62e9\u754c\u9762\u8bed\u8a00\u3002",
    },
}

FEATURE_LABELS_I18N = {
    "en": FEATURE_LABELS,
    "zh": {
        "age_at_baseline": "\u57fa\u7ebf\u5e74\u9f84",
        "sex": "\u6027\u522b",
        "demog_pteducat": "\u53d7\u6559\u80b2\u5e74\u9650",
        "demog_ptethcat": "\u65cf\u88d1\u7f16\u7801",
        "demog_ptraccat": "\u79cd\u65cf\u7f16\u7801",
        "mmse_mmscore": "MMSE",
        "adas_total13": "ADAS-Cog13",
        "faq_faqtotal": "FAQ \u603b\u5206",
        "ST130TA": "ST130TA\uff1a\u53f3\u989e\u4e2d\u56de\u76ae\u5c42\u539a\u5ea6",
        "ST88SV": "ST88SV\uff1a\u53f3\u6d77\u9a6c\u4f53\u79ef",
        "ST115CV": "ST115CV\uff1a\u53f3\u989d\u4e0a\u56de\u7070\u8d28\u4f53\u79ef",
        "ST99TA": "ST99TA\uff1a\u53f3\u5c9b\u53f6\u76ae\u5c42\u539a\u5ea6",
        "ST73CV": "ST73CV\uff1a\u53f3\u4fa7\u5c3e\u4fa7\u524d\u6263\u5e26\u7070\u8d28\u4f53\u79ef",
    },
}


st.set_page_config(
    page_title="ADNI 36-Month Runner",
    layout="wide",
    initial_sidebar_state="collapsed",
)


APP_CSS = """
<style>
    :root {
        --bg-a: #f6f9ff;
        --bg-b: #eef5ff;
        --bg-c: #eefaf7;
        --ink: #0f172a;
        --muted: #536076;
        --border: rgba(148, 163, 184, 0.18);
        --shadow: 0 24px 60px rgba(15, 23, 42, 0.10);
        --shadow-soft: 0 10px 26px rgba(15, 23, 42, 0.06);
        --blue: #2563eb;
        --teal: #14b8a6;
    }

    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(37, 99, 235, 0.14), transparent 24%),
            radial-gradient(circle at 100% 0%, rgba(20, 184, 166, 0.12), transparent 24%),
            radial-gradient(circle at 20% 100%, rgba(217, 119, 6, 0.10), transparent 22%),
            linear-gradient(180deg, var(--bg-a) 0%, var(--bg-b) 46%, var(--bg-c) 100%);
        color: var(--ink);
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    .hero-shell {
        display: grid;
        grid-template-columns: 1.45fr 0.78fr;
        gap: 1rem;
        padding: 1.25rem;
        border-radius: 28px;
        border: 1px solid rgba(255, 255, 255, 0.60);
        background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,255,255,0.72));
        box-shadow: var(--shadow);
        backdrop-filter: blur(14px);
        margin-bottom: 1rem;
    }

    .hero-copy h1 {
        margin: 0;
        font-size: 2rem;
        line-height: 1.06;
        letter-spacing: -0.03em;
        color: var(--ink);
    }

    .hero-copy p {
        margin: 0.6rem 0 0;
        max-width: 62ch;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.6;
    }

    .hero-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.95rem;
    }

    .hero-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.5rem 0.8rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.04);
        border: 1px solid rgba(15, 23, 42, 0.06);
        color: var(--ink);
        font-size: 0.9rem;
        font-weight: 600;
    }

    .hero-panel {
        border-radius: 22px;
        padding: 1rem;
        border: 1px solid rgba(37, 99, 235, 0.10);
        background:
            linear-gradient(180deg, rgba(37, 99, 235, 0.06), rgba(20, 184, 166, 0.05)),
            rgba(255, 255, 255, 0.85);
        box-shadow: var(--shadow-soft);
    }

    .panel-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #5b6b83;
        margin-bottom: 0.35rem;
    }

    .panel-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: var(--ink);
        margin-bottom: 0.45rem;
    }

    .panel-line {
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.55;
        margin-bottom: 0.5rem;
    }

    .panel-stat {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        padding: 0.7rem 0.8rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.70);
        border: 1px solid rgba(148, 163, 184, 0.18);
        margin-top: 0.5rem;
        font-size: 0.92rem;
        color: var(--ink);
    }

    .panel-stat strong {
        color: var(--blue);
    }

    .section-title {
        font-size: 1.08rem;
        font-weight: 800;
        color: var(--ink);
        margin: 0 0 0.35rem;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 0.94rem;
        margin: 0 0 0.8rem;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.76);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 0.85rem 0.9rem;
        box-shadow: var(--shadow-soft);
        backdrop-filter: blur(10px);
    }

    [data-testid="stForm"] {
        background: rgba(255,255,255,0.84);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 24px;
        padding: 1rem 1rem 1.1rem;
        box-shadow: var(--shadow-soft);
    }

    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.76);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        box-shadow: var(--shadow-soft);
    }

    [data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: var(--shadow-soft);
    }

    [data-testid="stDownloadButton"] button,
    [data-testid="stButton"] button,
    .stForm button {
        border-radius: 999px !important;
        border: 0 !important;
        background: linear-gradient(135deg, var(--blue), var(--teal)) !important;
        color: white !important;
        box-shadow: 0 14px 28px rgba(37, 99, 235, 0.20);
        font-weight: 700 !important;
    }

    [data-testid="stDownloadButton"] button:hover,
    [data-testid="stButton"] button:hover,
    .stForm button:hover {
        transform: translateY(-1px);
        box-shadow: 0 18px 30px rgba(37, 99, 235, 0.24);
    }

    [data-testid="stNumberInput"] input,
    [data-baseweb="select"] > div,
    [data-testid="stFileUploader"] {
        border-radius: 14px !important;
    }

    .viz-grid {
        display: grid;
        grid-template-columns: 1.05fr 0.95fr;
        gap: 1rem;
        margin-top: 0.8rem;
        margin-bottom: 1rem;
    }

    .viz-card {
        border-radius: 22px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: rgba(255, 255, 255, 0.84);
        box-shadow: var(--shadow-soft);
        padding: 1rem;
    }

    .viz-card-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #5b6b83;
        margin-bottom: 0.55rem;
        font-weight: 700;
    }

    .gauge-shell {
        display: grid;
        place-items: center;
        min-height: 320px;
    }

    .gauge-ring {
        width: 230px;
        height: 230px;
        border-radius: 50%;
        position: relative;
        display: grid;
        place-items: center;
        background: conic-gradient(var(--accent) 0 0%, rgba(148, 163, 184, 0.18) 0% 100%);
    }

    .gauge-ring::before {
        content: "";
        position: absolute;
        inset: 16px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.96);
        box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.10);
    }

    .gauge-center {
        position: relative;
        z-index: 1;
        text-align: center;
    }

    .gauge-value {
        font-size: 2.35rem;
        font-weight: 900;
        line-height: 1;
        letter-spacing: -0.04em;
        color: var(--ink);
    }

    .gauge-caption {
        margin-top: 0.45rem;
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--muted);
    }

    .band-track {
        position: relative;
        height: 22px;
        border-radius: 999px;
        overflow: hidden;
        background: rgba(226, 232, 240, 0.9);
        display: flex;
        margin-top: 0.7rem;
        box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.18);
    }

    .band-seg {
        height: 100%;
    }

    .band-seg.low {
        background: linear-gradient(90deg, #22c55e, #86efac);
    }

    .band-seg.mid {
        background: linear-gradient(90deg, #f59e0b, #fbbf24);
    }

    .band-seg.high {
        background: linear-gradient(90deg, #ef4444, #f87171);
    }

    .band-marker {
        position: absolute;
        top: -8px;
        width: 3px;
        height: 38px;
        border-radius: 999px;
        background: #0f172a;
        box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.8);
        transform: translateX(-1px);
    }

    .band-marker::after {
        content: "";
        position: absolute;
        top: 12px;
        left: 50%;
        transform: translateX(-50%);
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #0f172a;
        box-shadow: 0 0 0 4px rgba(15, 23, 42, 0.12);
    }

    .band-legend {
        display: flex;
        justify-content: space-between;
        gap: 0.5rem;
        margin-top: 0.7rem;
        font-size: 0.88rem;
        color: var(--muted);
    }

    .band-legend span {
        display: inline-flex;
        align-items: center;
        gap: 0.42rem;
    }

    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }

    .contrib-list {
        display: grid;
        gap: 0.78rem;
        margin-top: 0.55rem;
    }

    .contrib-header {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        align-items: center;
        flex-wrap: wrap;
        margin: 0.2rem 0 0.65rem;
    }

    .contrib-chip-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }

    .contrib-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.45rem 0.7rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: rgba(255, 255, 255, 0.84);
        color: var(--ink);
        box-shadow: var(--shadow-soft);
    }

    .contrib-chip strong {
        font-variant-numeric: tabular-nums;
    }

    .contrib-row {
        padding: 0.8rem 0.82rem 0.9rem;
        border-radius: 18px;
        background: rgba(248, 250, 252, 0.95);
        border: 1px solid rgba(148, 163, 184, 0.14);
    }

    .contrib-meta {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
        align-items: baseline;
    }

    .contrib-left {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        min-width: 0;
    }

    .contrib-rank {
        flex: 0 0 auto;
        padding: 0.22rem 0.5rem;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.10);
        color: var(--blue);
        font-size: 0.78rem;
        font-weight: 800;
    }

    .contrib-feature {
        font-weight: 800;
        color: var(--ink);
        font-size: 0.95rem;
        line-height: 1.35;
    }

    .contrib-value {
        font-size: 0.88rem;
        font-variant-numeric: tabular-nums;
        color: var(--muted);
        white-space: nowrap;
    }

    .contrib-track {
        display: grid;
        grid-template-columns: 1fr 2px 1fr;
        align-items: stretch;
        height: 16px;
        border-radius: 999px;
        background: rgba(226, 232, 240, 0.95);
        overflow: hidden;
        box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.12);
    }

    .contrib-rail {
        display: flex;
        align-items: stretch;
        overflow: hidden;
    }

    .contrib-rail.left {
        justify-content: flex-end;
    }

    .contrib-rail.right {
        justify-content: flex-start;
    }

    .contrib-center {
        background: rgba(15, 23, 42, 0.20);
    }

    .contrib-fill {
        height: 100%;
        min-width: 3px;
        border-radius: 999px;
    }

    .contrib-fill.pos {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.95), rgba(248, 113, 113, 0.85));
    }

    .contrib-fill.neg {
        background: linear-gradient(90deg, rgba(20, 184, 166, 0.85), rgba(34, 197, 94, 0.95));
    }

    .contrib-note {
        margin-top: 0.65rem;
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.45;
    }

    .contrib-empty {
        padding: 0.9rem 1rem;
        border-radius: 18px;
        background: rgba(248, 250, 252, 0.95);
        color: var(--muted);
        border: 1px dashed rgba(148, 163, 184, 0.28);
    }

    .impact-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.9rem;
        margin-top: 0.7rem;
    }

    .impact-col {
        border-radius: 22px;
        padding: 0.95rem;
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow: var(--shadow-soft);
    }

    .impact-head {
        display: flex;
        justify-content: space-between;
        gap: 0.7rem;
        align-items: center;
        margin-bottom: 0.8rem;
        flex-wrap: wrap;
    }

    .impact-title {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.95rem;
        font-weight: 800;
        color: var(--ink);
    }

    .impact-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }

    .impact-stat {
        font-size: 0.84rem;
        color: var(--muted);
        font-variant-numeric: tabular-nums;
    }

    .impact-list {
        display: grid;
        gap: 0.72rem;
    }

    .impact-item {
        padding: 0.75rem 0.78rem 0.82rem;
        border-radius: 16px;
        background: rgba(248, 250, 252, 0.94);
        border: 1px solid rgba(148, 163, 184, 0.12);
    }

    .impact-row {
        display: flex;
        justify-content: space-between;
        gap: 0.6rem;
        align-items: baseline;
        margin-bottom: 0.42rem;
    }

    .impact-name {
        font-weight: 800;
        color: var(--ink);
        font-size: 0.92rem;
        line-height: 1.35;
        min-width: 0;
    }

    .impact-value {
        font-size: 0.84rem;
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
        color: var(--muted);
    }

    .impact-bar {
        height: 12px;
        border-radius: 999px;
        overflow: hidden;
        background: rgba(226, 232, 240, 0.95);
        box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.10);
    }

    .impact-fill {
        height: 100%;
        border-radius: 999px;
    }

    .impact-fill.pos {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.95), rgba(251, 113, 133, 0.88));
    }

    .impact-fill.neg {
        background: linear-gradient(90deg, rgba(20, 184, 166, 0.95), rgba(34, 197, 94, 0.88));
    }

    .impact-empty {
        padding: 0.8rem 0.85rem;
        border-radius: 14px;
        background: rgba(248, 250, 252, 0.94);
        color: var(--muted);
        border: 1px dashed rgba(148, 163, 184, 0.24);
    }

    .hero-shell {
        position: relative;
        overflow: hidden;
        grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.82fr);
        gap: 1.2rem;
        padding: 1.55rem;
        border-radius: 34px;
        border: 1px solid rgba(255, 255, 255, 0.72);
        background:
            radial-gradient(circle at 10% 10%, rgba(37, 99, 235, 0.14), transparent 26%),
            radial-gradient(circle at 90% 0%, rgba(20, 184, 166, 0.16), transparent 24%),
            linear-gradient(135deg, rgba(255,255,255,0.96), rgba(246,250,255,0.82));
        box-shadow: 0 28px 72px rgba(15, 23, 42, 0.12);
    }

    .hero-shell::before {
        content: "";
        position: absolute;
        inset: auto -80px -120px auto;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(37, 99, 235, 0.16), rgba(37, 99, 235, 0.02) 64%, transparent 72%);
        pointer-events: none;
    }

    .hero-copy,
    .hero-panel {
        position: relative;
        z-index: 1;
    }

    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.42rem 0.78rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.05);
        border: 1px solid rgba(15, 23, 42, 0.08);
        color: #35506f;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
    }

    .hero-copy h1 {
        font-size: clamp(2.1rem, 4vw, 3.2rem);
        line-height: 0.98;
        max-width: 10.5ch;
    }

    .hero-copy p {
        max-width: 58ch;
        font-size: 1rem;
        line-height: 1.7;
    }

    .hero-chip-row {
        margin-top: 1.05rem;
        margin-bottom: 1.1rem;
    }

    .hero-chip {
        padding: 0.52rem 0.88rem;
        background: rgba(255, 255, 255, 0.76);
        border: 1px solid rgba(37, 99, 235, 0.10);
        box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
    }

    .summary-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.78rem;
        margin-top: 1.15rem;
    }

    .summary-card {
        display: grid;
        gap: 0.32rem;
        padding: 0.9rem 0.95rem;
        border-radius: 20px;
        background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(248,250,252,0.92));
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.05);
    }

    .summary-card span {
        color: var(--muted);
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .summary-card strong {
        color: var(--ink);
        font-size: 1rem;
        line-height: 1.35;
        letter-spacing: -0.01em;
    }

    .hero-panel-spotlight {
        position: relative;
        padding: 1.15rem;
        border-radius: 28px;
        border: 1px solid rgba(37, 99, 235, 0.12);
        background:
            radial-gradient(circle at 100% 0%, rgba(20, 184, 166, 0.16), transparent 36%),
            linear-gradient(180deg, rgba(37, 99, 235, 0.08), rgba(255,255,255,0.92));
        box-shadow: 0 22px 40px rgba(15, 23, 42, 0.08);
    }

    .hero-orb {
        width: 72px;
        height: 72px;
        border-radius: 24px;
        margin-bottom: 1rem;
        background:
            radial-gradient(circle at 30% 30%, rgba(255,255,255,0.92), transparent 36%),
            linear-gradient(135deg, rgba(37, 99, 235, 0.92), rgba(20, 184, 166, 0.9));
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.28), 0 16px 30px rgba(37, 99, 235, 0.18);
    }

    .panel-label {
        margin-bottom: 0.45rem;
    }

    .panel-title {
        font-size: 1.2rem;
        line-height: 1.25;
    }

    .panel-stat {
        background: rgba(255,255,255,0.82);
    }

    .form-shell {
        display: grid;
        gap: 1rem;
    }

    .form-head {
        display: grid;
        gap: 0.52rem;
        padding: 1rem 1.05rem;
        border-radius: 22px;
        background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(244,248,255,0.9));
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow: var(--shadow-soft);
    }

    .form-kicker {
        color: #3f5d80;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .form-title {
        color: var(--ink);
        font-size: 1.18rem;
        font-weight: 900;
        letter-spacing: -0.02em;
    }

    .form-note {
        color: var(--muted);
        font-size: 0.94rem;
        line-height: 1.6;
    }

    .form-pill-row {
        display: flex;
        gap: 0.55rem;
        flex-wrap: wrap;
        margin-top: 0.15rem;
    }

    .form-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.45rem 0.72rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.04);
        border: 1px solid rgba(15, 23, 42, 0.06);
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--ink);
    }

    .story-stack {
        display: grid;
        gap: 1rem;
    }

    .story-card {
        position: relative;
        overflow: hidden;
        padding: 1rem 1.05rem;
        border-radius: 24px;
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow: var(--shadow-soft);
    }

    .story-card::before {
        content: "";
        position: absolute;
        inset: 0 auto auto 0;
        width: 100%;
        height: 1px;
        background: linear-gradient(90deg, rgba(37, 99, 235, 0.28), rgba(20, 184, 166, 0.28), transparent);
    }

    .story-card-primary {
        background:
            radial-gradient(circle at 100% 0%, rgba(37, 99, 235, 0.10), transparent 30%),
            linear-gradient(180deg, rgba(255,255,255,0.92), rgba(245,250,255,0.9));
    }

    .story-label {
        color: #4d6786;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.45rem;
    }

    .story-title {
        color: var(--ink);
        font-size: 1.08rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        margin-bottom: 0.65rem;
    }

    .story-note {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.65;
        margin-bottom: 0.85rem;
    }

    .story-list {
        display: grid;
        gap: 0.75rem;
        margin-top: 0.8rem;
    }

    .story-step {
        display: grid;
        grid-template-columns: 44px 1fr;
        gap: 0.75rem;
        align-items: start;
        padding: 0.78rem 0.82rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.74);
        border: 1px solid rgba(148, 163, 184, 0.12);
    }

    .story-step span {
        display: grid;
        place-items: center;
        width: 44px;
        height: 44px;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.92), rgba(20, 184, 166, 0.92));
        color: white;
        font-size: 0.88rem;
        font-weight: 900;
        box-shadow: 0 12px 24px rgba(37, 99, 235, 0.16);
    }

    .story-step p {
        margin: 0;
        color: var(--ink);
        font-size: 0.92rem;
        line-height: 1.6;
        font-weight: 600;
    }

    .mini-cluster-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
    }

    .mini-cluster-card {
        display: grid;
        gap: 0.32rem;
        padding: 0.9rem 0.82rem;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(248,250,252,0.96), rgba(255,255,255,0.96));
        border: 1px solid rgba(148, 163, 184, 0.16);
    }

    .mini-cluster-card span {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        line-height: 1.35;
    }

    .mini-cluster-card strong {
        color: var(--ink);
        font-size: 1.28rem;
        font-weight: 900;
        letter-spacing: -0.03em;
    }

    .section-title {
        font-size: 1.15rem;
        margin-bottom: 0.42rem;
    }

    .section-subtitle {
        margin-bottom: 0.95rem;
        line-height: 1.6;
    }

    @media (max-width: 1100px) {
        .hero-shell,
        .viz-grid,
        .impact-grid {
            grid-template-columns: 1fr;
        }

        .summary-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 760px) {
        .hero-shell {
            padding: 1.15rem;
            border-radius: 26px;
        }

        .summary-grid,
        .mini-cluster-grid {
            grid-template-columns: 1fr;
        }

        .story-step {
            grid-template-columns: 40px 1fr;
        }

        .story-step span {
            width: 40px;
            height: 40px;
        }
    }
</style>
"""


def _load_runtime() -> dict[str, Any]:
    for path in ARTIFACT_CANDIDATES:
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError(
        "No model artifact found. Expected one of: "
        + ", ".join(str(path) for path in ARTIFACT_CANDIDATES)
    )


@st.cache_resource(show_spinner=False)
def load_runtime() -> dict[str, Any]:
    return _load_runtime()


def _lang_code() -> str:
    default_lang = st.session_state.get("lang", "en")
    left, right = st.columns([0.76, 0.24])
    with left:
        st.empty()
    with right:
        st.caption("Language / \u8bed\u8a00")
        choice = st.segmented_control(
            "Language",
            options=list(LANG_OPTIONS.keys()),
            default=default_lang,
            format_func=lambda code: LANG_OPTIONS[code],
            key="language_switch",
            label_visibility="collapsed",
        )
    lang = choice or default_lang
    st.session_state["lang"] = lang
    return lang


def t(lang: str, key: str, **kwargs: object) -> str:
    return TEXT[lang][key].format(**kwargs)


def feature_label(feature: str, lang: str) -> str:
    return FEATURE_LABELS_I18N[lang][feature]


def risk_group_label(group: str, lang: str) -> str:
    if lang == "zh":
        return {"low": "\u4f4e\u98ce\u9669", "intermediate": "\u4e2d\u98ce\u9669", "high": "\u9ad8\u98ce\u9669"}.get(group, group)
    return group.capitalize()


def months_text(lang: str, months: int | float) -> str:
    return f"{months} months" if lang == "en" else f"{months} \u4e2a\u6708"


def _hero_html(metadata: dict[str, Any], lang: str) -> str:
    perf = metadata["performance"]
    feature_count = len(metadata["features"])
    return dedent(
        f"""
        <div class="hero-shell">
          <div class="hero-copy">
            <div class="hero-eyebrow">{t(lang, "hero_eyebrow")}</div>
            <h1>{t(lang, "hero_title")}</h1>
            <p>{t(lang, "hero_subtitle")}</p>
            <div class="hero-chip-row">
              <span class="hero-chip">{t(lang, "hero_chip_offline")}</span>
              <span class="hero-chip">{t(lang, "hero_chip_artifact")}</span>
              <span class="hero-chip">{t(lang, "hero_chip_batch")}</span>
            </div>
            <div class="summary-grid">
              <div class="summary-card">
                <span>{t(lang, "model_label")}</span>
                <strong>{metadata["model_name"]}</strong>
              </div>
              <div class="summary-card">
                <span>{t(lang, "horizon_label")}</span>
                <strong>{months_text(lang, metadata["horizon_months"])}</strong>
              </div>
              <div class="summary-card">
                <span>{t(lang, "ribbon_auc")}</span>
                <strong>{perf["auc"]:.3f}</strong>
              </div>
              <div class="summary-card">
                <span>{t(lang, "ribbon_brier")}</span>
                <strong>{perf["brier"]:.3f}</strong>
              </div>
            </div>
          </div>
          <div class="hero-panel hero-panel-spotlight">
            <div class="hero-orb"></div>
            <div class="panel-label">{t(lang, "model_label")}</div>
            <div class="panel-title">{metadata["display_name"]}</div>
            <div class="panel-line">{t(lang, "local_artifact")}</div>
            <div class="panel-stat">
              <span>{t(lang, "ribbon_features")}</span>
              <strong>{feature_count}</strong>
            </div>
            <div class="panel-stat">
              <span>{t(lang, "ribbon_mode")}</span>
              <strong>{t(lang, "ribbon_mode_value")}</strong>
            </div>
          </div>
        </div>
        """
    ).strip()


def _overview_html(lang: str) -> str:
    feature_cards = "".join(
        dedent(
            f"""
            <div class="mini-cluster-card">
              <span>{t(lang, {'Demographics': 'group_demographics', 'Cognitive': 'group_cognitive', 'MRI': 'group_mri'}[group_name])}</span>
              <strong>{len(features)}</strong>
            </div>
            """
        ).strip()
        for group_name, features in FEATURE_GROUPS
    )
    return dedent(
        f"""
        <div class="story-stack">
          <div class="story-card story-card-primary">
            <div class="story-label">{t(lang, "workflow_title")}</div>
            <div class="story-title">{t(lang, "how_it_works")}</div>
            <div class="story-list">
              <div class="story-step"><span>01</span><p>{t(lang, "step_1")}</p></div>
              <div class="story-step"><span>02</span><p>{t(lang, "step_2")}</p></div>
              <div class="story-step"><span>03</span><p>{t(lang, "step_3")}</p></div>
            </div>
          </div>
          <div class="story-card">
            <div class="story-label">{t(lang, "feature_map_title")}</div>
            <div class="story-title">{t(lang, "single_case")}</div>
            <div class="story-note">{t(lang, "form_helper")}</div>
            <div class="mini-cluster-grid">{feature_cards}</div>
          </div>
        </div>
        """
    ).strip()


def _category_label(feature: str, value: str, lang: str) -> str:
    if feature == "sex":
        if lang == "zh":
            return {"Female": "\u5973", "Male": "\u7537"}.get(value, value)
        return value
    if lang == "zh":
        return f"{value} ({t(lang, 'training_code')})"
    return f"{value} ({t(lang, 'training_code')})"


def _render_widget(
    feature: str,
    defaults: dict[str, Any],
    widget_config: dict[str, dict[str, Any]],
    help_text: str,
    lang: str,
):
    if feature in CATEGORICAL_FEATURES:
        options = widget_config[feature]["options"]
        default_index = options.index(defaults[feature]) if defaults[feature] in options else 0
        return st.selectbox(
            feature_label(feature, lang),
            options=options,
            index=default_index,
            format_func=lambda option, f=feature: _category_label(f, option, lang),
            help=help_text,
        )

    cfg = widget_config[feature]
    step = float(cfg["step"])
    return st.number_input(
        feature_label(feature, lang),
        min_value=float(cfg["min"]),
        max_value=float(cfg["max"]),
        value=float(defaults[feature]),
        step=step,
        format="%.3f" if step < 1 else "%.1f",
        help=help_text,
    )


def _build_input_row(values: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for feature in metadata["features"]:
        value = values.get(feature, metadata["defaults"][feature])
        if pd.isna(value) or value == "":
            value = metadata["defaults"][feature]
        row[feature] = str(value) if feature in CATEGORICAL_FEATURES else float(value)
    return row


def score_batch(frame: pd.DataFrame, runtime: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    metadata = runtime["metadata"]
    defaults = metadata["defaults"]
    records: list[dict[str, Any]] = []
    missing_features = [feature for feature in metadata["features"] if feature not in frame.columns]

    for index, (_, row) in enumerate(frame.iterrows(), start=1):
        inputs: dict[str, Any] = {}
        for feature in metadata["features"]:
            value = row[feature] if feature in frame.columns else defaults[feature]
            if pd.isna(value) or value == "":
                value = defaults[feature]
            inputs[feature] = value

        result = score_inputs(inputs, runtime)
        participant_id = str(row["RID"]) if "RID" in frame.columns and pd.notna(row.get("RID")) else str(index)
        records.append(
            {
                "participant_id": participant_id,
                "probability": result["probability"],
                "probability_pct": result["probability"] * 100.0,
                "risk_group": result["risk_group"],
                "log_odds": result["log_odds"],
            }
        )

    return pd.DataFrame.from_records(records), missing_features


def _risk_theme(group: str) -> dict[str, str]:
    if group == "high":
        return {"accent": "#ef4444", "accent_soft": "rgba(239, 68, 68, 0.12)", "legend": "#ef4444"}
    if group == "intermediate":
        return {"accent": "#f59e0b", "accent_soft": "rgba(245, 158, 11, 0.12)", "legend": "#f59e0b"}
    return {"accent": "#22c55e", "accent_soft": "rgba(34, 197, 94, 0.12)", "legend": "#22c55e"}


def _visualize_prediction(
    lang: str,
    probability_pct: float,
    risk_group: str,
    thresholds: dict[str, float],
    log_odds: float,
) -> str:
    theme = _risk_theme(risk_group)
    p = max(0.0, min(100.0, probability_pct))
    low = max(0.0, min(100.0, thresholds["low_upper"] * 100.0))
    high = max(low, min(100.0, thresholds["high_lower"] * 100.0))
    mid = max(0.0, high - low)
    tail = max(0.0, 100.0 - high)
    track_style = f"background: conic-gradient({theme['accent']} 0 {p:.2f}%, rgba(148,163,184,0.16) {p:.2f}% 100%);"
    marker_left = max(0.0, min(100.0, p))
    return dedent(
        f"""
        <div class="viz-grid">
          <div class="viz-card">
            <div class="viz-card-label">{t(lang, "prediction")}</div>
            <div class="gauge-shell">
              <div class="gauge-ring" style="{track_style}">
                <div class="gauge-center">
                  <div class="gauge-value">{p:.1f}%</div>
                  <div class="gauge-caption">{risk_group_label(risk_group, lang)}</div>
                </div>
              </div>
            </div>
          </div>
          <div class="viz-card">
            <div class="viz-card-label">Risk band</div>
            <div class="panel-line">{t(lang, "thresholds", low=thresholds["low_upper"], high=thresholds["high_lower"])}</div>
            <div class="band-track">
              <div class="band-seg low" style="width:{low:.2f}%"></div>
              <div class="band-seg mid" style="width:{mid:.2f}%"></div>
              <div class="band-seg high" style="width:{tail:.2f}%"></div>
              <div class="band-marker" style="left:{marker_left:.2f}%"></div>
            </div>
            <div class="band-legend">
              <span><i class="legend-dot" style="background:#22c55e"></i>Low</span>
              <span><i class="legend-dot" style="background:#f59e0b"></i>Intermediate</span>
              <span><i class="legend-dot" style="background:#ef4444"></i>High</span>
            </div>
            <div class="hero-panel" style="margin-top:1rem; background:linear-gradient(180deg, {theme['accent_soft']}, rgba(255,255,255,0.86));">
              <div class="panel-label">Log-odds</div>
              <div class="panel-title">{log_odds:.3f}</div>
              <div class="panel-line">{t(lang, "risk_group")}: {risk_group_label(risk_group, lang)}</div>
            </div>
          </div>
        </div>
        """
    ).strip()


def _contribution_bars_html(contributions: pd.DataFrame, lang: str) -> str:
    if contributions.empty:
        return f'<div class="contrib-empty">{t(lang, "top_contributions")}: n/a</div>'

    values = contributions["contribution_log_odds"].astype(float).tolist()
    max_abs = max((abs(v) for v in values), default=0.0) or 1.0
    pos_count = sum(1 for v in values if v >= 0)
    neg_count = len(values) - pos_count
    pos_total = sum(abs(v) for v in values if v >= 0) or 1.0
    neg_total = sum(abs(v) for v in values if v < 0) or 1.0
    top_label = html.escape(str(contributions.iloc[0]["label"]))
    top_value = float(contributions.iloc[0]["contribution_log_odds"])
    pos_rows: list[str] = []
    neg_rows: list[str] = []

    positive = contributions[contributions["contribution_log_odds"] >= 0].copy()
    negative = contributions[contributions["contribution_log_odds"] < 0].copy()

    for rank, (_, row) in enumerate(positive.iterrows(), start=1):
        label = html.escape(str(row["label"]))
        value = float(row["contribution_log_odds"])
        share = abs(value) / pos_total * 100.0
        pos_rows.append(
            f"""
            <div class="impact-item">
              <div class="impact-row">
                <div class="impact-name">{rank}. {label}</div>
                <div class="impact-value">+{abs(value):.3f} {t(lang, "log_odds")} · {share:.1f}%</div>
              </div>
              <div class="impact-bar"><div class="impact-fill pos" style="width:{min(100.0, max(6.0, share)):.2f}%"></div></div>
            </div>
            """
        )

    for rank, (_, row) in enumerate(negative.iterrows(), start=1):
        label = html.escape(str(row["label"]))
        value = float(row["contribution_log_odds"])
        share = abs(value) / neg_total * 100.0
        neg_rows.append(
            f"""
            <div class="impact-item">
              <div class="impact-row">
                <div class="impact-name">{rank}. {label}</div>
                <div class="impact-value">-{abs(value):.3f} {t(lang, "log_odds")} · {share:.1f}%</div>
              </div>
              <div class="impact-bar"><div class="impact-fill neg" style="width:{min(100.0, max(6.0, share)):.2f}%"></div></div>
            </div>
            """
        )

    summary = f"""
    <div class="contrib-header">
      <div class="contrib-chip-row">
        <span class="contrib-chip"><strong>{pos_count}</strong> {t(lang, "contrib_positive")}</span>
        <span class="contrib-chip"><strong>{neg_count}</strong> {t(lang, "contrib_negative")}</span>
      </div>
      <div class="contrib-chip-row">
        <span class="contrib-chip">{t(lang, "contrib_top")}: <strong>{top_label}</strong></span>
        <span class="contrib-chip">{top_value:+.3f}</span>
      </div>
    </div>
    """
    body = f"""
    <div class="impact-grid">
      <div class="impact-col">
        <div class="impact-head">
          <div class="impact-title"><span class="impact-dot" style="background:#ef4444"></span>{t(lang, "contrib_positive")}</div>
          <div class="impact-stat">{sum(1 for v in values if v >= 0)} {t(lang, "contrib_items")} · {sum(abs(v) for v in values if v >= 0):.3f}</div>
        </div>
        <div class="impact-list">
          {''.join(pos_rows) if pos_rows else f'<div class="impact-empty">{t(lang, "contrib_positive")}: n/a</div>'}
        </div>
      </div>
      <div class="impact-col">
        <div class="impact-head">
          <div class="impact-title"><span class="impact-dot" style="background:#14b8a6"></span>{t(lang, "contrib_negative")}</div>
          <div class="impact-stat">{sum(1 for v in values if v < 0)} {t(lang, "contrib_items")} · {sum(abs(v) for v in values if v < 0):.3f}</div>
        </div>
        <div class="impact-list">
          {''.join(neg_rows) if neg_rows else f'<div class="impact-empty">{t(lang, "contrib_negative")}: n/a</div>'}
        </div>
      </div>
    </div>
    """
    note = f'<div class="contrib-note">{t(lang, "contrib_note")}</div>'
    return dedent(f"{summary}{body}{note}").strip()


runtime = load_runtime()
metadata = runtime["metadata"]
defaults = metadata["defaults"]
widget_config = metadata["widget_config"]
lang = _lang_code()

st.markdown(APP_CSS, unsafe_allow_html=True)

st.markdown(_hero_html(metadata, lang), unsafe_allow_html=True)

left, right = st.columns([1.12, 0.88], gap="large")

with left:
    with st.container(border=True):
        st.markdown(
            dedent(
                f"""
                <div class="form-shell">
                  <div class="form-head">
                    <div class="form-kicker">{t(lang, "single_case")}</div>
                    <div class="form-title">{t(lang, "prediction")}</div>
                    <div class="form-note">{t(lang, "single_case_hint")}</div>
                    <div class="form-pill-row">
                      <span class="form-pill">{t(lang, "group_demographics")} 5</span>
                      <span class="form-pill">{t(lang, "group_cognitive")} 3</span>
                      <span class="form-pill">{t(lang, "group_mri")} 5</span>
                    </div>
                  </div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )

        with st.form("single_case_form"):
            values: dict[str, Any] = {}
            for index, (group_name, features) in enumerate(FEATURE_GROUPS):
                group_key = {
                    "Demographics": "group_demographics",
                    "Cognitive": "group_cognitive",
                    "MRI": "group_mri",
                }[group_name]
                st.markdown(f"**{t(lang, group_key)}**")
                st.caption(", ".join(feature_label(feature, lang) for feature in features))
                cols = st.columns(2)
                for idx, feature in enumerate(features):
                    with cols[idx % 2]:
                        values[feature] = _render_widget(
                            feature,
                            defaults,
                            widget_config,
                            metadata["feature_help"][feature],
                            lang,
                        )
                if index != len(FEATURE_GROUPS) - 1:
                    st.divider()
            submitted = st.form_submit_button(t(lang, "predict_button"), use_container_width=True)

with right:
    with st.container(border=True):
        st.markdown(_overview_html(lang), unsafe_allow_html=True)

if submitted:
    result = score_inputs(_build_input_row(values, metadata), runtime)
    probability_pct = result["probability"] * 100.0
    thresholds = metadata["risk_group_thresholds"]

    st.divider()
    with st.container(border=True):
        st.markdown(f'<div class="section-title">{t(lang, "prediction")}</div>', unsafe_allow_html=True)
        st.markdown(
            _visualize_prediction(
                lang,
                probability_pct,
                result["risk_group"],
                thresholds,
                result["log_odds"],
            ),
            unsafe_allow_html=True,
        )

        result_cols = st.columns(3)
        result_cols[0].metric(t(lang, "probability"), f"{probability_pct:.1f}%")
        result_cols[1].metric(t(lang, "risk_group"), risk_group_label(result["risk_group"], lang))
        result_cols[2].metric(t(lang, "log_odds"), f"{result['log_odds']:.3f}")

        contribution_df = pd.DataFrame(result["contributions"])
        if not contribution_df.empty:
            contribution_df = contribution_df.copy()
            contribution_df["abs_contribution"] = contribution_df["contribution_log_odds"].abs()
            contribution_df["label"] = contribution_df["feature"].map(lambda feature: feature_label(feature, lang))
            contribution_df = contribution_df.sort_values("abs_contribution", ascending=False).head(6)
            st.markdown(f"**{t(lang, 'top_contributions')}**")
            st.markdown(_contribution_bars_html(contribution_df, lang), unsafe_allow_html=True)
        else:
            st.info(t(lang, "top_contributions"))

        with st.expander(t(lang, "entered_values"), expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {t(lang, "feature"): feature_label(feature, lang), t(lang, "value"): result["input_row"][feature]}
                        for feature in metadata["features"]
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

with st.container(border=True):
    with st.expander(t(lang, "batch_title"), expanded=False):
        st.markdown(f'<div class="section-subtitle">{t(lang, "batch_desc")}</div>', unsafe_allow_html=True)
        template_df = pd.DataFrame([{feature: defaults[feature] for feature in metadata["features"]}])
        st.download_button(
            t(lang, "download_template"),
            template_df.to_csv(index=False).encode("utf-8"),
            file_name="adni_minimal_template.csv",
            mime="text/csv",
        )

        uploaded = st.file_uploader(t(lang, "upload_csv"), type=["csv"], key="minimal_batch_uploader")
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            st.dataframe(batch_df.head(10), use_container_width=True, hide_index=True)

            if st.button(t(lang, "score_csv"), type="primary"):
                scored_df, missing_features = score_batch(batch_df, runtime)
                if missing_features:
                    st.warning(t(lang, "missing_columns", cols=", ".join(missing_features)))
                display_scored = scored_df.rename(
                    columns={
                        "participant_id": t(lang, "participant_id"),
                        "probability": t(lang, "probability"),
                        "probability_pct": f"{t(lang, 'probability')} (%)",
                        "risk_group": t(lang, "risk_group"),
                        "log_odds": t(lang, "log_odds"),
                    }
                )
                st.dataframe(display_scored, use_container_width=True, hide_index=True)
                st.download_button(
                    t(lang, "download_scored"),
                    scored_df.to_csv(index=False).encode("utf-8"),
                    file_name="adni_minimal_scored.csv",
                    mime="text/csv",
                )

with st.container(border=True):
    with st.expander(t(lang, "model_note"), expanded=False):
        if lang == "zh":
            st.write(
                "\u8be5\u8ba1\u7b97\u5668\u9075\u5faa ADNI \u5185\u90e8\u6700\u7ec8\u6a21\u578b\u5b9a\u4e49\uff0c\u9002\u5408\u7814\u7a76\u5c55\u793a\u4e0e\u65b9\u6cd5\u6f14\u793a\uff0c\u4e0d\u5efa\u8bae\u4f5c\u4e3a\u81ea\u52a8\u8bca\u65ad\u5de5\u5177\u3002"
            )
        else:
            st.write(metadata["model_note"])
        st.code(
            json.dumps(
                {
                    "model_name": metadata["model_name"],
                    "horizon_months": metadata["horizon_months"],
                    "features": metadata["features"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            language="json",
        )
