#!/usr/bin/env python3
"""Minimal Streamlit page for offline deployment of the 36-month calculator."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --- 路径与模块配置 ---
WEBAPP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEBAPP_DIR.parent
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

# 确保 model_runtime 存在且能正常导入
from model_runtime import CATEGORICAL_FEATURES, FEATURE_LABELS, score_inputs

# --- 常量与配置 ---
ARTIFACT_CANDIDATES = [
    PROJECT_ROOT / "artifacts" / "final_model_36m_model2_lr.joblib",
    PROJECT_ROOT / "results" / "web_calculator_36m_model2_lr_v1" / "artifacts" / "final_model_36m_model2_lr.joblib",
]

FEATURE_GROUPS = [
    ("Demographics", ["age_at_baseline", "sex", "demog_pteducat", "demog_ptethcat", "demog_ptraccat"]),
    ("Cognitive", ["mmse_mmscore", "adas_total13", "faq_faqtotal"]),
    ("MRI", ["ST130TA", "ST88SV", "ST115CV", "ST99TA", "ST73CV"]),
]

GROUP_ICONS = {
    "Demographics": "📋",
    "Cognitive": "🧠",
    "MRI": "🩻"
}

LANG_OPTIONS = {"en": "English", "zh": "中文"}

# 优化的深色调配色方案 (增强视觉冲击力和清晰度)
COLOR_PALETTE = {
    "low": "#2C7A7B",         # Deep Teal (深青色 - 低风险)
    "intermediate": "#D97706", # Deep Amber (深琥珀 - 中风险)
    "high": "#C53030",        # Deep Red (深红色 - 高风险)
    "background": "rgba(0,0,0,0)",
    "grid_line": "#E2E8F0",
    "slate": "#475569"        # Deep Slate (深板岩灰)
}

TEXT = {
    "en": {
        "hero_title": "ADNI 36-Month Risk Calculator",
        "single_case": "Single Case Assessment",
        "single_case_hint": "Please provide the patient's baseline characteristics below.",
        "predict_button": "Run Prediction",
        "how_it_works": "System Information",
        "step_1": "1. Loads the exported joblib artifact without external dependencies.",
        "step_2": "2. Converts inputs into the optimized model schema.",
        "step_3": "3. Returns probability, risk stratification, and feature contributions.",
        "prediction": "Clinical Evaluation Report",
        "clinical_summary": "Automated Clinical Summary",
        "probability": "Probability",
        "risk_group": "Risk Stratification",
        "log_odds": "Log-odds",
        "top_contributions": "Risk Drivers & Protective Factors",
        "batch_title": "Batch CSV Scoring",
        "batch_desc": "Upload a standard CSV file to evaluate multiple participants simultaneously.",
        "download_template": "Download CSV Template",
        "upload_csv": "Upload Patient Data",
        "score_csv": "Start Batch Scoring",
        "download_scored": "Download Results CSV",
        "missing_columns": "Missing columns were imputed using training defaults: {cols}",
        "group_demographics": "Demographics",
        "group_cognitive": "Cognitive",
        "group_mri": "MRI",
        "training_code": "code",
        "local_artifact": "Local artifact: artifacts/final_model_36m_model2_lr.joblib",
    },
    "zh": {
        "hero_title": "ADNI 36个月疾病进展预测系统",
        "single_case": "单病例录入评估",
        "single_case_hint": "请在此处完整输入患者的基线数据，随后点击底部按钮获取分析报告。",
        "predict_button": "生成预测报告",
        "how_it_works": "系统运行机制",
        "step_1": "1. 零依赖加载本地导出的 joblib 模型核心。",
        "step_2": "2. 将表单数据静默转换为模型标准特征空间。",
        "step_3": "3. 实时推断疾病概率、风险分层及特征影响力。",
        "prediction": "临床评估综合报告",
        "clinical_summary": "AI 自动化研判结论",
        "probability": "进展概率",
        "risk_group": "风险分层",
        "log_odds": "对数几率 (Log-odds)",
        "top_contributions": "病理驱动与保护因素溯源",
        "batch_title": "CSV 队列数据批量评估",
        "batch_desc": "上传包含标准化特征列的数据表格，一键完成大批量队列数据的评分推断。",
        "download_template": "下载空白特征模板",
        "upload_csv": "上传队列数据 (CSV)",
        "score_csv": "开始批量推断",
        "download_scored": "下载评估结果报告",
        "missing_columns": "由于缺少部分列，已自动使用基线常模数据补齐：{cols}",
        "group_demographics": "人口学及社会特征",
        "group_cognitive": "神经心理学量表",
        "group_mri": "结构性磁共振 (sMRI)",
        "training_code": "编码",
        "local_artifact": "本地模型路径：artifacts/final_model_36m_model2_lr.joblib",
    },
}

FEATURE_LABELS_I18N = {
    "en": FEATURE_LABELS,
    "zh": {
        "age_at_baseline": "基线年龄 (岁)",
        "sex": "生理性别",
        "demog_pteducat": "受教育年限 (年)",
        "demog_ptethcat": "族裔背景",
        "demog_ptraccat": "种族背景",
        "mmse_mmscore": "MMSE 简明精神状态",
        "adas_total13": "ADAS-Cog13 总分",
        "faq_faqtotal": "FAQ 认知功能活动",
        "ST130TA": "ST130TA：右侧颞中回皮层厚度",
        "ST88SV": "ST88SV：右侧海马体积",
        "ST115CV": "ST115CV：右侧额上回灰质体积",
        "ST99TA": "ST99TA：右侧岛叶皮层厚度",
        "ST73CV": "ST73CV：右侧尾侧前扣带体积",
    },
}

# --- 初始化与样式 ---
st.set_page_config(page_title="ADNI 36-Month Risk Assessor", layout="wide", initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
    /* 1. 网页整体背景：极淡的蓝灰色，提供视觉衬托 */
    [data-testid="stAppViewContainer"] {{
        background-color: #F0F4F8; 
    }}
    
    /* 隐藏顶部默认的透明渐变装饰条，使界面更干净 */
    [data-testid="stHeader"] {{
        background-color: transparent;
    }}

    /* 2. 主体容器：中央悬浮的白色大卡片 */
    .block-container {{
        max-width: 1080px !important;
        background-color: #FFFFFF;
        padding: 3rem 4rem 4rem 4rem !important; 
        margin-top: 3rem;
        margin-bottom: 4rem;
        border-radius: 16px;
        border: 1px solid #E2E8F0; 
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01); 
    }}
    
    /* 警告框去底色 */
    div[data-testid="stAlert"] {{
        background-color: transparent !important;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
    }}
    
    /* === 强制生效：强化三组特征卡片的背景与边框 === */
    /* 容器设为等高伸展 */
    [data-testid="stForm"] [data-testid="stHorizontalBlock"] {{
        align-items: stretch !important;
    }}
    
    /* 统一定义所有特征列的卡片内边距和圆角，并强制生效 */
    [data-testid="stForm"] [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
        padding: 1.5rem !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        margin-bottom: 1rem !important;
    }}
    
    /* 第一列：人口学特征 (主题色：板岩灰) */
    [data-testid="stForm"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(1) {{
        border: 2px solid #CBD5E1 !important;
        background-color: #F8FAFC !important;
    }}
    
    /* 第二列：认知特征 (主题色：深青色) */
    [data-testid="stForm"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) {{
        border: 2px solid #9FD6D6 !important;
        background-color: #F0FDFD !important;
    }}
    
    /* 第三列：MRI特征 (主题色：深红色) */
    [data-testid="stForm"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(3) {{
        border: 2px solid #FEB2B2 !important;
        background-color: #FFF5F5 !important;
    }}
    /* ============================================== */
    
    /* 优化表单提交按钮 */
    div[data-testid="stForm"] button[kind="primary"] {{
        border-radius: 8px !important;
        font-weight: bold !important;
        padding-top: 0.75rem !important;
        padding-bottom: 0.75rem !important;
    }}
    
    .metric-box {{
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }}
</style>
""", unsafe_allow_html=True)

def _load_runtime() -> dict[str, Any]:
    for path in ARTIFACT_CANDIDATES:
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError("未找到模型工件，请检查路径。")

@st.cache_resource(show_spinner=False)
def load_runtime() -> dict[str, Any]:
    return _load_runtime()

def t(lang: str, key: str, **kwargs: object) -> str:
    return TEXT[lang][key].format(**kwargs)

def feature_label(feature: str, lang: str) -> str:
    return FEATURE_LABELS_I18N[lang].get(feature, feature)

def risk_group_label(group: str, lang: str) -> str:
    if lang == "zh":
        return {"low": "低风险组", "intermediate": "中度风险组", "high": "高风险组"}.get(group, group)
    return group.capitalize()

def get_risk_hex_color(group: str) -> str:
    return COLOR_PALETTE.get(group, "#CBD5E1")

def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

def _render_widget(feature: str, defaults: dict[str, Any], widget_config: dict[str, dict[str, Any]], help_text: str, lang: str):
    label = feature_label(feature, lang)
    if feature in CATEGORICAL_FEATURES:
        options = widget_config[feature]["options"]
        default_index = options.index(defaults[feature]) if defaults[feature] in options else 0
        def format_func(opt):
            if feature == "sex" and lang == "zh":
                return {"Female": "女性", "Male": "男性"}.get(opt, opt)
            return f"{opt} ({t(lang, 'training_code')})"
        return st.selectbox(label, options=options, index=default_index, format_func=format_func, help=help_text)

    cfg = widget_config[feature]
    step = float(cfg["step"])
    return st.number_input(label, min_value=float(cfg["min"]), max_value=float(cfg["max"]), value=float(defaults[feature]), step=step, format="%.3f" if step < 1 else "%.1f", help=help_text)

def _build_input_row(values: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    return {f: str(values.get(f, metadata["defaults"][f])) if f in CATEGORICAL_FEATURES else float(values.get(f, metadata["defaults"][f])) for f in metadata["features"]}

def score_batch(frame: pd.DataFrame, runtime: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    metadata = runtime["metadata"]
    defaults = metadata["defaults"]
    records, missing_features = [], [f for f in metadata["features"] if f not in frame.columns]

    for index, (_, row) in enumerate(frame.iterrows(), start=1):
        inputs = {f: row[f] if f in frame.columns and pd.notna(row[f]) else defaults[f] for f in metadata["features"]}
        result = score_inputs(inputs, runtime)
        records.append({
            "participant_id": str(row.get("RID", index)),
            "probability_pct": result["probability"] * 100.0,
            "risk_group": result["risk_group"],
            "log_odds": result["log_odds"],
        })
    return pd.DataFrame.from_records(records), missing_features

# --- 主程序执行 ---
runtime = load_runtime()
metadata = runtime["metadata"]
defaults = metadata["defaults"]
widget_config = metadata["widget_config"]

# 顶部 Header 区域
col_title, col_lang = st.columns([0.88, 0.12], vertical_alignment="top")
with col_lang:
    lang = st.selectbox("Language", options=["zh", "en"], index=0, label_visibility="collapsed")

with col_title:
    st.markdown(f"## {t(lang, 'hero_title')}")

st.markdown("<br>", unsafe_allow_html=True)

# --- 单例表单输入区 ---
st.markdown(f"#### ❖ {t(lang, 'single_case')}")
st.markdown(f"<p style='color: #64748B; margin-bottom: 1.5rem;'>{t(lang, 'single_case_hint')}</p>", unsafe_allow_html=True)

with st.form("single_case_form", border=False):
    values: dict[str, Any] = {}
    group_cols = st.columns(3, gap="large")
    
    for idx, (group_name, features) in enumerate(FEATURE_GROUPS):
        group_key = {"Demographics": "group_demographics", "Cognitive": "group_cognitive", "MRI": "group_mri"}[group_name]
        icon = GROUP_ICONS[group_name]
        
        with group_cols[idx]:
            st.markdown(f"<h5 style='color: #1E293B; margin-bottom: 1.25rem;'>{icon} {t(lang, group_key)}</h5>", unsafe_allow_html=True)
            for f in features:
                values[f] = _render_widget(f, defaults, widget_config, metadata["feature_help"].get(f, ""), lang)
                
    st.markdown("<br>", unsafe_allow_html=True)
    # 取消底部的 st.columns，直接铺满一个加宽的预测按钮，避免干扰前面列容器的选择器
    submitted = st.form_submit_button(t(lang, "predict_button"), type="primary", use_container_width=True)

# --- 预测结果展示区 ---
if submitted:
    st.divider()
    
    input_row = _build_input_row(values, metadata)
    result = score_inputs(input_row, runtime)
    prob_pct = result["probability"] * 100.0
    r_group = result["risk_group"]
    
    st.markdown(f"#### ❖ {t(lang, 'prediction')}")
    
    # 提取主要特征贡献
    contrib_df = pd.DataFrame(result["contributions"])
    top_risk, top_prot = None, None
    if not contrib_df.empty:
        pos_mask = contrib_df['contribution_log_odds'] > 0
        neg_mask = contrib_df['contribution_log_odds'] < 0
        if pos_mask.any():
            top_risk = feature_label(contrib_df.loc[contrib_df[pos_mask]['contribution_log_odds'].idxmax()]['feature'], lang)
        if neg_mask.any():
            top_prot = feature_label(contrib_df.loc[contrib_df[neg_mask]['contribution_log_odds'].idxmin()]['feature'], lang)

    # 动态生成研判文本
    risk_label = risk_group_label(r_group, lang)
    if lang == "zh":
        summary_text = f"经系统综合研判，该患者当前被划分为<strong style='color:{get_risk_hex_color(r_group)}'>【{risk_label}】</strong>，36个月内疾病进展的推断概率为 <strong>{prob_pct:.1f}%</strong>。"
        if top_risk: summary_text += f" 其致险推力主要来源于 <strong>{top_risk}</strong> 的异常表现；"
        if top_prot: summary_text += f" 而 <strong>{top_prot}</strong> 则提供了主要的保护性对抗作用。"
    else:
        summary_text = f"Based on the assessment, this patient is classified as <strong style='color:{get_risk_hex_color(r_group)}'>[{risk_label}]</strong> with an estimated 36-month progression probability of <strong>{prob_pct:.1f}%</strong>."
        if top_risk: summary_text += f" The primary risk-increasing driver is <strong>{top_risk}</strong>;"
        if top_prot: summary_text += f" The leading protective factor is <strong>{top_prot}</strong>."

    # 渲染动态 Banner 
    bg_color = hex_to_rgba(get_risk_hex_color(r_group), 0.12)
    border_color = get_risk_hex_color(r_group)
    icon = "🔴" if r_group == "high" else "🟡" if r_group == "intermediate" else "🟢"
    
    st.markdown(f"""
    <div style="padding: 1.25rem 1.5rem; border-radius: 12px; background-color: {bg_color}; border-left: 6px solid {border_color}; margin-top: 1rem; margin-bottom: 2rem;">
        <h5 style="margin-top: 0; color: #1E293B; margin-bottom: 0.75rem;">{icon} {t(lang, 'clinical_summary')}</h5>
        <p style="font-size: 1.05rem; color: #334155; margin-bottom: 0; line-height: 1.6;">{summary_text}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 图表双栏布局
    res_col1, res_col2 = st.columns([1, 1.6], gap="large")
    
    with res_col1:
        # Plotly 仪表盘
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={'suffix': "%", 'font': {'size': 42, 'color': '#1E293B'}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
                'bar': {'color': get_risk_hex_color(r_group), 'thickness': 0.75},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 33], 'color': hex_to_rgba(COLOR_PALETTE['low'], 0.35)},        
                    {'range': [33, 66], 'color': hex_to_rgba(COLOR_PALETTE['intermediate'], 0.35)}, 
                    {'range': [66, 100], 'color': hex_to_rgba(COLOR_PALETTE['high'], 0.35)}
                ]
            }
        ))
        fig_gauge.update_layout(
            height=260, 
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor=COLOR_PALETTE["background"],
            plot_bgcolor=COLOR_PALETTE["background"]
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
        
        # 底部指标框
        m_col1, m_col2 = st.columns(2)
        m_col1.markdown(f"<div class='metric-box'><div style='color:#64748b;font-size:0.85rem;margin-bottom:0.25rem;'>{t(lang,'probability')}</div><div style='font-size:1.3rem;font-weight:700;color:#1E293B;'>{prob_pct:.1f}%</div></div>", unsafe_allow_html=True)
        m_col2.markdown(f"<div class='metric-box'><div style='color:#64748b;font-size:0.85rem;margin-bottom:0.25rem;'>{t(lang,'log_odds')}</div><div style='font-size:1.3rem;font-weight:700;color:#1E293B;'>{result['log_odds']:.3f}</div></div>", unsafe_allow_html=True)

    with res_col2:
        if not contrib_df.empty:
            st.markdown(f"<h6 style='color: #475569; text-align: center; margin-bottom: 0;'>{t(lang, 'top_contributions')}</h6>", unsafe_allow_html=True)
            
            contrib_df["display_label"] = contrib_df["feature"].apply(
                lambda f: f"{feature_label(f, lang)} = {input_row.get(f, 'N/A')}"
            )
            
            contrib_df["abs_val"] = contrib_df["contribution_log_odds"].abs()
            contrib_df = contrib_df.sort_values("abs_val", ascending=True).tail(7) 
            
            contrib_df['color'] = contrib_df['contribution_log_odds'].apply(
                lambda x: COLOR_PALETTE["high"] if x > 0 else COLOR_PALETTE["low"]
            )
            
            fig_bar = go.Figure(go.Bar(
                x=contrib_df['contribution_log_odds'],
                y=contrib_df['display_label'],
                orientation='h',
                marker=dict(
                    color=contrib_df['color'],
                    line=dict(color='rgba(255, 255, 255, 0.8)', width=1.5) 
                ),
                opacity=1.0, 
                text=contrib_df['contribution_log_odds'].apply(lambda x: f"{x:+.3f}"),
                textposition='outside', 
                textfont=dict(color='#1E293B', size=13, weight='bold'),
                hovertemplate="<b>%{y}</b><br>贡献值: %{x:+.3f}<extra></extra>"
            ))
            
            fig_bar.update_layout(
                height=340, 
                margin=dict(l=10, r=50, t=20, b=10), 
                xaxis=dict(
                    showgrid=False, 
                    zeroline=True, 
                    zerolinecolor='#94A3B8', 
                    zerolinewidth=2,
                    showticklabels=False 
                ),
                yaxis=dict(showgrid=False, tickfont=dict(color='#1E293B', size=12)),
                showlegend=False,
                paper_bgcolor=COLOR_PALETTE["background"],
                plot_bgcolor=COLOR_PALETTE["background"]
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

# --- 批量评估模块 ---
st.divider()
st.markdown(f"#### ❖ {t(lang, 'batch_title')}")
st.markdown(f"<p style='color: #64748B; margin-bottom: 1.5rem;'>{t(lang, 'batch_desc')}</p>", unsafe_allow_html=True)

col_dl, col_up = st.columns([1, 2], gap="large")
with col_dl:
    st.markdown("<br>", unsafe_allow_html=True)
    template_df = pd.DataFrame([{f: defaults[f] for f in metadata["features"]}])
    st.download_button(t(lang, "download_template"), template_df.to_csv(index=False).encode("utf-8"), "template.csv", "text/csv", use_container_width=True)
    
with col_up:
    uploaded = st.file_uploader(t(lang, "upload_csv"), type=["csv"], label_visibility="collapsed")
    
if uploaded is not None:
    batch_df = pd.read_csv(uploaded)
    if st.button(t(lang, "score_csv"), type="primary"):
        scored_df, missing_cols = score_batch(batch_df, runtime)
        if missing_cols:
            st.info(t(lang, "missing_columns", cols=", ".join(missing_cols)))
        
        st.dataframe(scored_df, use_container_width=True)
        st.download_button(t(lang, "download_scored"), scored_df.to_csv(index=False).encode("utf-8"), "scored_results.csv", "text/csv")

# --- 底部：系统说明模块 ---
st.divider()
st.markdown(f"<h6 style='color: #64748B;'>{t(lang, 'how_it_works')}</h6>", unsafe_allow_html=True)
st.markdown(f"""
<div style='color: #64748B; font-size: 0.85rem; line-height: 1.6;'>
{t(lang, 'step_1')}<br>
{t(lang, 'step_2')}<br>
{t(lang, 'step_3')}<br><br>
💾 {t(lang, 'local_artifact')}
</div>
""", unsafe_allow_html=True)