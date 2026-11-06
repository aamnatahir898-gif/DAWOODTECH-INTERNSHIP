import base64
import io
import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parent
if not (PROJECT_DIR / "models" / "final_model.joblib").exists():
    if (PROJECT_DIR.parent / "models" / "final_model.joblib").exists():
        PROJECT_DIR = PROJECT_DIR.parent

BASE_DIR = PROJECT_DIR
METRICS_PATH = BASE_DIR / "results" / "final_test_metrics.json"
SHAP_PATH = BASE_DIR / "results" / "shap_feature_importance.csv"

VIDEO_URL = ""
THRESHOLD = 0.58
MEDIUM_RISK = 0.30

DEEP = "#04242A"
TEAL = "#0E5E63"
BRIGHT = "#2EC4B6"
GOLD = "#E9C46A"
SILVER = "#D7E3E8"
CORAL = "#FF8A7A"
TEXT = "#EAF6F5"
MUTED = "#9FC7C4"

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}
FEATURES = NUMERIC_FEATURES + list(CATEGORICAL_OPTIONS.keys())

NAV_HOME = "🏠  Home"
NAV_PREDICT = "🔮  Predict"
NAV_BATCH = "📂  Batch"
NAV_INSIGHTS = "📊  Insights"
NAV_ITEMS = [NAV_HOME, NAV_PREDICT, NAV_BATCH, NAV_INSIGHTS]


def build_hero_svg():
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 420" preserveAspectRatio="xMidYMid slice">',
        '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#04242A"/><stop offset="1" stop-color="#14605E"/>'
        "</linearGradient></defs>",
        '<rect width="1200" height="420" fill="url(#g)"/>',
    ]
    nodes = []
    for i in range(12):
        for j in range(4):
            x = 60 + i * 95 + 18 * math.sin(i * 1.7)
            y = 50 + j * 100 + 22 * math.cos(i * 0.9 + j)
            nodes.append((x, y))
    for index, (x, y) in enumerate(nodes):
        i, j = divmod(index, 4)
        if j < 3:
            x2, y2 = nodes[index + 1]
            parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#2EC4B6" stroke-opacity="0.22" stroke-width="1.4"/>')
        if i < 11:
            x2, y2 = nodes[index + 4]
            parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#2EC4B6" stroke-opacity="0.16" stroke-width="1.2"/>')
    for index, (x, y) in enumerate(nodes):
        if index % 7 == 0:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#E9C46A" fill-opacity="0.85"/>')
        else:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="#2EC4B6" fill-opacity="0.7"/>')
    parts.append("</svg>")
    return "".join(parts)


HERO_IMAGE = "data:image/svg+xml;base64," + base64.b64encode(build_hero_svg().encode("utf-8")).decode("utf-8")

CSS = """
<style>
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes shine {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(255,138,122,0.55); }
    70% { box-shadow: 0 0 0 10px rgba(255,138,122,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,138,122,0); }
}
@keyframes drift {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}

.stApp {
    background: linear-gradient(165deg, __DEEP__ 0%, #0A3F45 50%, #14605E 100%);
    color: __TEXT__;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1280px; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #021A1F 0%, #05282E 100%);
    border-right: 1px solid rgba(215,227,232,0.10);
}
[data-testid="stSidebar"] * { color: __TEXT__; }
.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 4px 22px 4px;
    border-bottom: 1px solid rgba(215,227,232,0.10);
    margin-bottom: 18px;
}
.brand-icon {
    width: 44px;
    height: 44px;
    border-radius: 14px;
    background: linear-gradient(135deg, __BRIGHT__, __GOLD__);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    animation: drift 5s ease-in-out infinite;
}
.brand-name { font-size: 20px; font-weight: 800; color: #FFFFFF; }
.brand-sub { font-size: 11px; color: __MUTED__; letter-spacing: 1.4px; }
div[role="radiogroup"] { gap: 6px; }
div[role="radiogroup"] > label {
    padding: 11px 14px;
    border-radius: 12px;
    border: 1px solid transparent;
    transition: background 0.2s ease, border-color 0.2s ease;
    width: 100%;
}
div[role="radiogroup"] > label:hover {
    background: rgba(46,196,182,0.08);
}
div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(46,196,182,0.16);
    border-color: rgba(46,196,182,0.45);
}
div[role="radiogroup"] > label > div:first-child { display: none; }
div[role="radiogroup"] > label p { font-size: 15px; font-weight: 600; color: __TEXT__ !important; }
.side-card {
    margin-top: 22px;
    padding: 16px;
    border-radius: 14px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(215,227,232,0.10);
    font-size: 12px;
    color: __MUTED__;
    line-height: 1.8;
}

h1, h2, h3, h4, p, span, label, li { color: __TEXT__; }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: __MUTED__ !important; }

[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(215,227,232,0.12);
    border-radius: 16px;
}
[data-testid="stExpander"] summary p { color: __SILVER__ !important; font-weight: 700; }
div[data-baseweb="select"] > div, .stNumberInput input {
    background: #0A3A40 !important;
    color: __TEXT__ !important;
    border-color: rgba(215,227,232,0.18) !important;
    border-radius: 10px !important;
}

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 26px;
    padding: 58px 48px;
    background-image: linear-gradient(120deg, rgba(4,36,42,0.90), rgba(14,94,99,0.72)), url("__HERO__");
    background-size: cover;
    background-position: center;
    border: 1px solid rgba(215,227,232,0.14);
    box-shadow: 0 18px 40px rgba(0,0,0,0.30);
    animation: fadeUp 0.8s ease both;
    margin-bottom: 26px;
}
.eyebrow {
    color: __BRIGHT__;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.hero-title {
    font-size: 48px;
    font-weight: 800;
    line-height: 1.08;
    background: linear-gradient(90deg, #FFFFFF, __SILVER__, __GOLD__, #FFFFFF);
    background-size: 250% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 7s linear infinite;
    margin: 0 0 16px 0;
}
.hero-sub { color: #DDF3F1; font-size: 17px; max-width: 640px; line-height: 1.6; margin: 0; }
.pill-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 24px; }
.pill {
    padding: 9px 18px;
    border-radius: 999px;
    background: rgba(46,196,182,0.14);
    border: 1px solid rgba(46,196,182,0.40);
    color: __TEXT__;
    font-size: 14px;
    font-weight: 600;
}

.banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    padding: 26px 30px;
    border-radius: 22px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(215,227,232,0.12);
    animation: fadeUp 0.7s ease both;
    margin-bottom: 26px;
}
.banner-title { font-size: 22px; font-weight: 800; color: #FFFFFF; margin-bottom: 6px; }
.banner-text { font-size: 14px; color: __MUTED__; line-height: 1.7; }
.flow { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.flow-node {
    padding: 12px 16px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(14,94,99,0.9), rgba(46,196,182,0.35));
    border: 1px solid rgba(215,227,232,0.20);
    color: #FFFFFF;
    font-size: 13px;
    font-weight: 700;
    animation: drift 4s ease-in-out infinite;
}
.flow-arrow { color: __GOLD__; font-size: 18px; font-weight: 800; }

.feature {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(215,227,232,0.12);
    border-radius: 20px;
    padding: 26px;
    min-height: 200px;
    animation: fadeUp 0.7s ease both;
    transition: transform 0.25s ease, border-color 0.25s ease;
}
.feature:hover { transform: translateY(-6px); border-color: __BRIGHT__; }
.feature-icon { font-size: 30px; }
.feature-title { color: #FFFFFF; font-size: 19px; font-weight: 800; margin: 12px 0 8px 0; }
.feature-text { color: __MUTED__; font-size: 14px; line-height: 1.65; }

.stat {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(215,227,232,0.16);
    border-radius: 18px;
    padding: 22px 24px;
    animation: fadeUp 0.6s ease both;
    transition: transform 0.25s ease, border-color 0.25s ease;
}
.stat:hover { transform: translateY(-4px); border-color: __BRIGHT__; }
.stat-label { color: __MUTED__; font-size: 12px; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase; }
.stat-value {
    font-size: 36px;
    font-weight: 800;
    margin-top: 8px;
    background: linear-gradient(90deg, __SILVER__, __GOLD__, __BRIGHT__);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-note { color: __MUTED__; font-size: 12px; margin-top: 4px; }

.steps {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(215,227,232,0.12);
    border-radius: 20px;
    padding: 26px;
    animation: fadeUp 0.7s ease both;
}
.step { display: flex; gap: 14px; align-items: flex-start; margin-bottom: 16px; }
.step-num {
    min-width: 34px;
    height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg, __BRIGHT__, __GOLD__);
    color: __DEEP__;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
}
.step-title { color: #FFFFFF; font-weight: 700; font-size: 15px; }
.step-text { color: __MUTED__; font-size: 13px; margin-top: 2px; line-height: 1.6; }

.section-title {
    color: __BRIGHT__;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.page-title { font-size: 32px; font-weight: 800; color: #FFFFFF; margin-bottom: 4px; }
.page-sub { color: __MUTED__; margin-bottom: 18px; font-size: 15px; }

.panel {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(215,227,232,0.10);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 16px;
    animation: fadeUp 0.6s ease both;
}
.result {
    text-align: center;
    background: linear-gradient(135deg, rgba(14,94,99,0.92), rgba(10,63,69,0.92));
    border: 1px solid rgba(233,196,106,0.35);
    border-radius: 22px;
    padding: 26px;
    animation: fadeUp 0.5s ease both;
}
.result-number {
    font-size: 56px;
    font-weight: 800;
    background: linear-gradient(90deg, __SILVER__, __GOLD__);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.risk-pill {
    display: inline-block;
    margin-top: 14px;
    padding: 8px 20px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 13px;
    letter-spacing: 1px;
    color: __DEEP__;
}
.risk-high { background: __CORAL__; animation: pulse 2s infinite; }
.risk-medium { background: __GOLD__; }
.risk-low { background: __BRIGHT__; }
.reason {
    background: rgba(255,255,255,0.04);
    border-left: 4px solid __BRIGHT__;
    border-radius: 12px;
    padding: 12px 16px;
    margin: 10px 0;
    animation: fadeUp 0.5s ease both;
}
.reason-title { color: #FFFFFF; font-weight: 700; font-size: 14px; }
.reason-text { color: __MUTED__; font-size: 12px; margin-top: 3px; }

.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    background: linear-gradient(90deg, #1B8A87, __BRIGHT__);
    color: __DEEP__;
    border: none;
    border-radius: 12px;
    font-weight: 800;
    padding: 10px 22px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(46,196,182,0.35);
    color: __DEEP__;
}
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border: 1px dashed rgba(46,196,182,0.5);
    border-radius: 14px;
}
footer { visibility: hidden; }
</style>
"""

for token, value in {
    "__DEEP__": DEEP,
    "__TEAL__": TEAL,
    "__BRIGHT__": BRIGHT,
    "__GOLD__": GOLD,
    "__SILVER__": SILVER,
    "__CORAL__": CORAL,
    "__TEXT__": TEXT,
    "__MUTED__": MUTED,
    "__HERO__": HERO_IMAGE,
}.items():
    CSS = CSS.replace(token, value)

st.set_page_config(
    page_title="ChurnGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)


def html(markup):
    cleaned = " ".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def stat(label, value, note):
    html(f"""
        <div class="stat">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-note">{note}</div>
        </div>
    """)


def style_fig(fig, height):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        colorway=[BRIGHT, GOLD, SILVER, CORAL],
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def humanize(name):
    return name.replace("cat__", "").replace("num__", "").replace("_", " ")


def risk_level(probability):
    if probability >= THRESHOLD:
        return "High", "risk-high"
    if probability >= MEDIUM_RISK:
        return "Medium", "risk-medium"
    return "Low", "risk-low"


def go_to(page_name):
    st.session_state["nav"] = page_name


def save_prediction(record):
    st.session_state.history.append(record)


@st.cache_resource
def load_model():
    preprocessor = joblib.load(BASE_DIR / "models" / "final_preprocessor.joblib")
    classifier = XGBClassifier()
    classifier.load_model(BASE_DIR / "models" / "final_classifier.json")
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as file:
            return json.load(file)
    return {}


@st.cache_data
def load_global_importance():
    if SHAP_PATH.exists():
        return pd.read_csv(SHAP_PATH).head(10)
    return pd.DataFrame()


def get_shap_values(classifier, transformed):
    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(transformed)
    if hasattr(values, "values"):
        values = values.values
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def explain_customer(model, customer):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed = preprocessor.transform(customer)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()
    transformed = pd.DataFrame(transformed, columns=feature_names)
    values = get_shap_values(classifier, transformed)[0]

    explanation = pd.DataFrame({"Feature": feature_names, "Contribution": values})
    positive = explanation[explanation["Contribution"] > 0]
    if positive.empty:
        positive = explanation.copy()
    return positive.sort_values("Contribution", ascending=False).head(3)


def pdf_table(rows, widths):
    table = Table(rows, colWidths=widths, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B7D3D0")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0E5E63")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def build_prediction_pdf(details, probability, risk, verdict, reasons):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("ChurnGuard Prediction Report", styles["Title"]),
        Paragraph("Generated on " + datetime.now().strftime("%d %B %Y, %H:%M"), styles["Normal"]),
        Spacer(1, 14),
        Paragraph("Prediction summary", styles["Heading2"]),
        pdf_table([
            ["Item", "Value"],
            ["Churn probability", f"{probability:.1%}"],
            ["Risk level", risk],
            ["Prediction", verdict],
            ["Decision threshold", f"{THRESHOLD:.2f}"],
        ], [190, 300]),
        Spacer(1, 12),
        Paragraph("Main factors in this score", styles["Heading2"]),
        pdf_table([["Factor", "Effect on churn risk"]] + [[f, d] for f, d in reasons], [190, 300]),
        Spacer(1, 12),
        Paragraph("Customer details", styles["Heading2"]),
        pdf_table([["Field", "Value"]] + [[k, str(v)] for k, v in details.items()], [190, 300]),
        Spacer(1, 14),
        Paragraph("SHAP explanations describe model behaviour and do not prove causes.", styles["Normal"]),
    ]
    doc.build(story)
    return buffer.getvalue()


def build_batch_pdf(results):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    total = len(results)
    churn = int((results["Prediction"] == "Churn").sum())
    high = int((results["Risk_Level"] == "High").sum())
    average = float(results["Churn_Probability"].mean())

    top = results.sort_values("Churn_Probability", ascending=False).head(15)
    rows = [["Tenure", "Contract", "Monthly charges", "Probability", "Risk"]]
    for _, row in top.iterrows():
        rows.append([
            str(int(row["tenure"])),
            str(row["Contract"]),
            f"${row['MonthlyCharges']:.2f}",
            f"{row['Churn_Probability']:.1%}",
            str(row["Risk_Level"]),
        ])

    story = [
        Paragraph("ChurnGuard Batch Summary", styles["Title"]),
        Paragraph("Generated on " + datetime.now().strftime("%d %B %Y, %H:%M"), styles["Normal"]),
        Spacer(1, 14),
        Paragraph("Overview", styles["Heading2"]),
        pdf_table([
            ["Item", "Value"],
            ["Customers analyzed", f"{total:,}"],
            ["Predicted churn", f"{churn:,}"],
            ["High risk customers", f"{high:,}"],
            ["Average churn probability", f"{average:.1%}"],
            ["Decision threshold", f"{THRESHOLD:.2f}"],
        ], [220, 270]),
        Spacer(1, 12),
        Paragraph("Top 15 customers by churn probability", styles["Heading2"]),
        pdf_table(rows, [70, 130, 110, 100, 90]),
    ]
    doc.build(story)
    return buffer.getvalue()


try:
    model = load_model()
except Exception as error:
    st.error(f"Model could not be loaded: {error}")
    st.stop()

metrics = load_metrics()
global_importance = load_global_importance()


def metric(key, default):
    return float(metrics.get(key, default))


ACCURACY = metric("accuracy", 0.7764)
PRECISION = metric("precision", 0.5601)
RECALL = metric("recall", 0.7353)
F1 = metric("f1_score", 0.6358)
ROC_AUC = metric("roc_auc", 0.8483)
PR_AUC = metric("pr_auc", 0.6710)

if "history" not in st.session_state:
    st.session_state.history = []
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None
if "nav" not in st.session_state:
    st.session_state.nav = NAV_HOME

with st.sidebar:
    html("""
        <div class="brand">
            <div class="brand-icon">🛡️</div>
            <div>
                <div class="brand-name">ChurnGuard</div>
                <div class="brand-sub">RETENTION INTELLIGENCE</div>
            </div>
        </div>
    """)
    page = st.radio("Navigation", NAV_ITEMS, key="nav", label_visibility="collapsed")
    html(f"""
        <div class="side-card">
            <b style="color:#FFFFFF;">Model</b><br>
            XGBoost Weighted<br><br>
            <b style="color:#FFFFFF;">Decision threshold</b><br>
            {THRESHOLD:.2f}<br><br>
            <b style="color:#FFFFFF;">Explanations</b><br>
            SHAP TreeExplainer
        </div>
    """)


if page == NAV_HOME:
    html("""
        <div class="hero">
            <div class="eyebrow">Customer retention platform</div>
            <div class="hero-title">Predict customer churn before it happens</div>
            <p class="hero-sub">ChurnGuard estimates the chance that a customer will leave,
            explains the main reasons behind each score, and helps you decide who should receive
            a retention offer.</p>
            <div class="pill-row">
                <span class="pill">Predict churn</span>
                <span class="pill">Explain each score</span>
                <span class="pill">Plan retention offers</span>
            </div>
        </div>
    """)

    st.button("Start prediction", on_click=go_to, args=(NAV_PREDICT,))
    st.write("")

    if VIDEO_URL:
        st.video(VIDEO_URL)
    else:
        html("""
            <div class="banner">
                <div>
                    <div class="banner-title">How ChurnGuard works</div>
                    <div class="banner-text">Customer data goes into a trained model, which returns a churn
                    probability. SHAP then shows which features pushed the score up or down.</div>
                </div>
                <div class="flow">
                    <span class="flow-node">Customer data</span>
                    <span class="flow-arrow">→</span>
                    <span class="flow-node">XGBoost model</span>
                    <span class="flow-arrow">→</span>
                    <span class="flow-node">Churn probability</span>
                    <span class="flow-arrow">→</span>
                    <span class="flow-node">Retention decision</span>
                </div>
            </div>
        """)

    a, b, c = st.columns(3)
    with a:
        stat("PR-AUC", f"{PR_AUC:.1%}", "Main metric for imbalanced churn data")
    with b:
        stat("ROC-AUC", f"{ROC_AUC:.1%}", "How well customers are ranked by risk")
    with c:
        stat("Recall", f"{RECALL:.1%}", f"Churners identified at a {THRESHOLD:.0%} threshold")

    st.write("")
    f1_col, f2_col, f3_col = st.columns(3)
    with f1_col:
        html("""
            <div class="feature">
                <div class="feature-icon">🔮</div>
                <div class="feature-title">Predict</div>
                <div class="feature-text">Enter a customer's details and get a churn probability
                with a clear risk level, updated instantly as you type.</div>
            </div>
        """)
    with f2_col:
        html("""
            <div class="feature" style="animation-delay:0.1s;">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">Explain</div>
                <div class="feature-text">See the top factors behind each score, using SHAP values
                from the trained model.</div>
            </div>
        """)
    with f3_col:
        html("""
            <div class="feature" style="animation-delay:0.2s;">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">Decide</div>
                <div class="feature-text">Run batch analysis on a CSV file, then download a
                CSV or PDF report for your retention team.</div>
            </div>
        """)

    st.write("")
    left, right = st.columns([1, 1], gap="large")
    with left:
        html("""
            <div class="steps">
                <div class="section-title">Getting started</div>
                <div class="step">
                    <div class="step-num">1</div>
                    <div>
                        <div class="step-title">Open Predict</div>
                        <div class="step-text">Fill in the customer profile, services and billing details.</div>
                    </div>
                </div>
                <div class="step">
                    <div class="step-num">2</div>
                    <div>
                        <div class="step-title">Review the result</div>
                        <div class="step-text">Check the probability, risk level and the main factors.</div>
                    </div>
                </div>
                <div class="step">
                    <div class="step-num">3</div>
                    <div>
                        <div class="step-title">Export</div>
                        <div class="step-text">Download a PDF report for a single customer, or a CSV for many.</div>
                    </div>
                </div>
            </div>
        """)
    with right:
        html('<div class="section-title">Model performance</div>')
        perf_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
            "Score": [ACCURACY, PRECISION, RECALL, F1, ROC_AUC, PR_AUC],
        })
        fig = px.bar(perf_df, x="Metric", y="Score", text=perf_df["Score"].map(lambda v: f"{v:.1%}"))
        fig.update_traces(marker_color=BRIGHT, textposition="outside")
        fig.update_layout(yaxis_range=[0, 1], xaxis_title="", yaxis_title="")
        st.plotly_chart(style_fig(fig, 320), width="stretch")


elif page == NAV_PREDICT:
    html("""
        <div class="page-title">Customer risk prediction</div>
        <div class="page-sub">Change any value below. The result, reasons and details update immediately.</div>
    """)

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        with st.expander("Customer profile", expanded=True):
            p1, p2 = st.columns(2)
            with p1:
                senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                gender = st.selectbox("Gender", CATEGORICAL_OPTIONS["gender"])
                partner = st.selectbox("Partner", CATEGORICAL_OPTIONS["Partner"])
                dependents = st.selectbox("Dependents", CATEGORICAL_OPTIONS["Dependents"])
            with p2:
                tenure = st.number_input("Tenure (months)", 0, 72, 12)
                monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=1.0)
                total = st.number_input(
                    "Total charges ($)", 0.0, 10000.0, float(round(monthly * tenure, 2)), step=10.0
                )

        with st.expander("Services", expanded=True):
            s1, s2 = st.columns(2)
            with s1:
                phone = st.selectbox("Phone service", CATEGORICAL_OPTIONS["PhoneService"])
                multiple = st.selectbox("Multiple lines", CATEGORICAL_OPTIONS["MultipleLines"])
                internet = st.selectbox("Internet service", CATEGORICAL_OPTIONS["InternetService"])
                security = st.selectbox("Online security", CATEGORICAL_OPTIONS["OnlineSecurity"])
                backup = st.selectbox("Online backup", CATEGORICAL_OPTIONS["OnlineBackup"])
            with s2:
                protection = st.selectbox("Device protection", CATEGORICAL_OPTIONS["DeviceProtection"])
                tech = st.selectbox("Tech support", CATEGORICAL_OPTIONS["TechSupport"])
                tv = st.selectbox("Streaming TV", CATEGORICAL_OPTIONS["StreamingTV"])
                movies = st.selectbox("Streaming movies", CATEGORICAL_OPTIONS["StreamingMovies"])

        with st.expander("Account and billing", expanded=True):
            b1, b2 = st.columns(2)
            with b1:
                contract = st.selectbox("Contract", CATEGORICAL_OPTIONS["Contract"])
                paperless = st.selectbox("Paperless billing", CATEGORICAL_OPTIONS["PaperlessBilling"])
            with b2:
                payment = st.selectbox("Payment method", CATEGORICAL_OPTIONS["PaymentMethod"])

        html('<div class="section-title" style="margin-top:14px;">Your entered details</div>')

    customer = pd.DataFrame([{
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
    }])[FEATURES]

    probability = float(model.predict_proba(customer)[0][1])
    risk, risk_class = risk_level(probability)
    verdict = "Churn" if probability >= THRESHOLD else "No churn"
    reasons_df = explain_customer(model, customer)
    reasons = [
        (humanize(row["Feature"]), "Increases churn risk" if float(row["Contribution"]) > 0 else "Reduces churn risk")
        for _, row in reasons_df.iterrows()
    ]
    details = {col: customer.iloc[0][col] for col in FEATURES}

    with left:
        st.dataframe(
            pd.DataFrame({"Field": FEATURES, "Value": [str(details[col]) for col in FEATURES]}),
            width="stretch",
            hide_index=True,
            height=300,
        )

    with right:
        html(f"""
            <div class="result">
                <div class="section-title">Churn probability</div>
                <div class="result-number">{probability:.1%}</div>
                <span class="risk-pill {risk_class}">{risk.upper()} RISK</span>
                <div style="color:{MUTED};margin-top:14px;font-size:14px;">
                    Prediction: <b style="color:#FFFFFF;">{verdict}</b>
                </div>
            </div>
        """)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"color": TEXT}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": MUTED},
                "bar": {"color": BRIGHT},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(46,196,182,0.25)"},
                    {"range": [30, 58], "color": "rgba(233,196,106,0.25)"},
                    {"range": [58, 100], "color": "rgba(255,138,122,0.28)"},
                ],
                "threshold": {"line": {"color": "#FFFFFF", "width": 3}, "value": THRESHOLD * 100},
            },
        ))
        gauge.update_layout(
            height=230,
            margin=dict(l=20, r=20, t=20, b=5),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color=TEXT,
        )
        st.plotly_chart(gauge, width="stretch")

        html('<div class="section-title">Why this score?</div>')
        for feature, direction in reasons:
            html(f"""
                <div class="reason">
                    <div class="reason-title">{feature}</div>
                    <div class="reason-text">{direction}</div>
                </div>
            """)
        html('<div class="reason-text" style="margin:8px 0 14px 0;">SHAP values explain how the model reached this score. They do not prove causes.</div>')

        pdf_bytes = build_prediction_pdf(details, probability, risk, verdict, reasons)
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download PDF report",
                data=pdf_bytes,
                file_name="churnguard_prediction.pdf",
                mime="application/pdf",
            )
        with d2:
            st.button(
                "Save prediction",
                on_click=save_prediction,
                args=({
                    "Tenure": tenure,
                    "Contract": contract,
                    "Monthly charges": monthly,
                    "Probability": f"{probability:.1%}",
                    "Risk": risk,
                },),
            )

    if st.session_state.history:
        st.write("")
        html('<div class="section-title">Saved predictions</div>')
        st.dataframe(pd.DataFrame(st.session_state.history), width="stretch", hide_index=True)


elif page == NAV_BATCH:
    html("""
        <div class="page-title">Batch analysis</div>
        <div class="page-sub">Upload a CSV with many customers to score them all at once.</div>
    """)

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    html(f"""
        <div style="font-size:13px;color:{MUTED};margin:6px 0 16px 0;">
            Required: the 19 model input columns. The customerID and Churn columns are ignored if present.
        </div>
    """)

    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            batch = batch.drop(columns=["customerID", "Churn", "ChurnFlag"], errors="ignore")
            missing = [col for col in FEATURES if col not in batch.columns]

            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            elif batch.empty:
                st.warning("The CSV has no customer records.")
            else:
                batch = batch[FEATURES].copy()
                for col in NUMERIC_FEATURES:
                    batch[col] = pd.to_numeric(batch[col], errors="coerce")

                html('<div class="section-title">Data preview</div>')
                st.dataframe(batch.head(10), width="stretch", hide_index=True)

                if batch.isnull().any().any():
                    st.error("Some values are missing or invalid. Please clean the CSV and upload it again.")
                elif st.button("Run batch analysis"):
                    with st.spinner("Scoring customers..."):
                        probabilities = model.predict_proba(batch)[:, 1]
                        results = batch.copy()
                        results["Churn_Probability"] = np.round(probabilities, 4)
                        results["Prediction"] = np.where(probabilities >= THRESHOLD, "Churn", "No Churn")
                        results["Risk_Level"] = [risk_level(p)[0] for p in probabilities]
                        st.session_state.batch_results = results
        except Exception as error:
            st.error(f"The CSV could not be read: {error}")

    results = st.session_state.batch_results
    if results is not None:
        st.write("")
        a, b, c = st.columns(3)
        with a:
            stat("Customers analyzed", f"{len(results):,}", "Records in this file")
        with b:
            stat("Predicted churn", f"{int((results['Prediction'] == 'Churn').sum()):,}", f"Above {THRESHOLD:.2f} threshold")
        with c:
            stat("High risk", f"{int((results['Risk_Level'] == 'High').sum()):,}", "Risk level: High")

        st.write("")
        st.dataframe(results, width="stretch", hide_index=True, height=360)

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download CSV report",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="churnguard_predictions.csv",
                mime="text/csv",
            )
        with d2:
            st.download_button(
                "Download PDF summary",
                data=build_batch_pdf(results),
                file_name="churnguard_batch_summary.pdf",
                mime="application/pdf",
            )

        chart_data = results["Risk_Level"].value_counts().reset_index()
        chart_data.columns = ["Risk Level", "Customers"]
        fig = px.pie(chart_data, names="Risk Level", values="Customers", hole=0.6)
        fig.update_traces(marker=dict(colors=[CORAL, GOLD, BRIGHT]))
        st.plotly_chart(style_fig(fig, 320), width="stretch")


elif page == NAV_INSIGHTS:
    html("""
        <div class="page-title">Model insights</div>
        <div class="page-sub">Performance on the held-out test set and the strongest churn indicators.</div>
    """)

    a, b, c, d = st.columns(4)
    with a:
        stat("ROC-AUC", f"{ROC_AUC:.2%}", "Test dataset")
    with b:
        stat("PR-AUC", f"{PR_AUC:.2%}", "Test dataset")
    with c:
        stat("Recall", f"{RECALL:.2%}", f"At {THRESHOLD:.0%} threshold")
    with d:
        stat("Precision", f"{PRECISION:.2%}", "Share of offers that reach churners")

    st.write("")
    html('<div class="section-title">Global feature importance</div>')
    if not global_importance.empty:
        importance = global_importance.copy()
        importance["Feature"] = importance["Feature"].apply(humanize)
        fig = px.bar(
            importance.sort_values("Mean_Absolute_SHAP"),
            x="Mean_Absolute_SHAP",
            y="Feature",
            orientation="h",
        )
        fig.update_traces(marker_color=BRIGHT)
        fig.update_layout(xaxis_title="Mean |SHAP value|", yaxis_title="")
        st.plotly_chart(style_fig(fig, 420), width="stretch")
    else:
        st.info("The SHAP feature importance file was not found.")

    html(f"""
        <div class="panel">
            <div class="section-title">Model configuration</div>
            <div style="color:{TEXT};line-height:2;">
                Algorithm: XGBoost Classifier<br>
                Training approach: Class-weighted learning<br>
                Decision threshold: {THRESHOLD:.2f}<br>
                Explainability: SHAP TreeExplainer<br>
                Evaluation: Stratified 5-fold cross-validation and a held-out test set
            </div>
        </div>
    """)