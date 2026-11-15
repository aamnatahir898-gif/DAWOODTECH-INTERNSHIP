import base64
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from joblib import load
from scipy.sparse import load_npz

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

REQUIRED = [
    RESULTS_DIR / "segment_cards.json",
    RESULTS_DIR / "customer_recommendations.parquet",
    RESULTS_DIR / "popularity_top10.csv",
    RESULTS_DIR / "item_similar_products.parquet",
    RESULTS_DIR / "recommender_metrics.json",
    RESULTS_DIR / "k_selection.csv",
    RESULTS_DIR / "clustering_comparison.json",
    RESULTS_DIR / "split_info.json",
    RESULTS_DIR / "products.parquet",
    MODELS_DIR / "clustering_models.joblib",
    MODELS_DIR / "item_similarity.npz",
    MODELS_DIR / "catalog_items.parquet",
    MODELS_DIR / "scoring_config.json",
    PROCESSED_DIR / "rfm_clustered.pkl",
]

SEGMENT_ORDER = ["Champions", "Loyal", "Potential Loyalist", "New", "At Risk", "Lost"]
SEGMENT_STYLE = {
    "Champions": ("#00e0c6", "🏆"),
    "Loyal": ("#7bd389", "💚"),
    "Potential Loyalist": ("#ffd166", "🌱"),
    "New": ("#9d8cff", "✨"),
    "At Risk": ("#ff7a59", "⚠️"),
    "Lost": ("#8b93a7", "💤"),
}
SEGMENT_ACTIONS = {
    "Champions": "Early access, VIP rewards and referral invites",
    "Loyal": "Cross-sell related categories, loyalty points and bundles",
    "Potential Loyalist": "Second-purchase incentive and personalized picks",
    "New": "Welcome series, review request and second-order discount",
    "At Risk": "Win-back email and previously purchased products",
    "Lost": "Low-cost reactivation email and suppress from paid ads",
}
METHOD_LABEL = {
    "popularity": "Most popular",
    "item_cf": "Bought together",
    "als": "ALS factor model",
}
MAX_PER_CATEGORY = 2


st.set_page_config(page_title="ShopSegments", page_icon="🛒", layout="wide")


def background_uri():
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='1600' height='1000'>"
        "<defs><filter id='b'><feGaussianBlur stdDeviation='95'/></filter></defs>"
        "<rect width='1600' height='1000' fill='#0b1020'/>"
        "<circle cx='260' cy='190' r='300' fill='#7c5cff' opacity='.55' filter='url(#b)'/>"
        "<circle cx='1360' cy='170' r='250' fill='#00e0c6' opacity='.38' filter='url(#b)'/>"
        "<circle cx='1120' cy='880' r='340' fill='#ff7a59' opacity='.32' filter='url(#b)'/>"
        "<circle cx='260' cy='900' r='260' fill='#3a7bd5' opacity='.35' filter='url(#b)'/>"
        "</svg>"
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


CSS = """
<style>
@keyframes fadeUp {from {opacity:0; transform:translateY(16px);} to {opacity:1; transform:none;}}
@keyframes grow {from {width:0;} to {width:var(--w);}}
.stApp {background:url("__BG__") center/cover fixed no-repeat,#0b1020 !important; color:#eef2ff;}
header[data-testid="stHeader"] {background:transparent !important;}
.block-container {padding-top:2rem; max-width:1280px;}
h1,h2,h3,h4,p,label,span,div {color:#eef2ff;}
.brand {display:flex; align-items:center; gap:12px; animation:fadeUp .6s ease both; margin-bottom:18px;}
.brand .logo {width:44px; height:44px; border-radius:14px; display:flex; align-items:center; justify-content:center;
  background:linear-gradient(135deg,#7c5cff,#00e0c6); font-size:22px;}
.brand .name {font-size:1.6rem; font-weight:800;}
.glass {background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.12); backdrop-filter:blur(14px);
  border-radius:18px; padding:18px; position:relative; overflow:hidden; box-shadow:0 10px 30px rgba(0,0,0,.25);
  animation:fadeUp .6s ease both; transition:transform .25s, box-shadow .25s;}
.glass:hover {transform:translateY(-4px); box-shadow:0 18px 40px rgba(0,0,0,.35);}
.glass .accent {position:absolute; left:0; top:0; bottom:0; width:5px;}
.kpis {display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:22px;}
.kpi .label {color:#a9b1c6; font-size:.8rem; text-transform:uppercase; letter-spacing:.8px;}
.kpi .value {font-size:1.9rem; font-weight:800; margin-top:4px;
  background:linear-gradient(90deg,#00e0c6,#7c5cff); -webkit-background-clip:text; color:transparent;}
.grid3 {display:grid; grid-template-columns:repeat(3,1fr); gap:18px; margin-bottom:10px;}
.grid-prod {display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:16px;}
.big {font-size:2.1rem; font-weight:800; margin:8px 0 2px;}
.muted {color:#a9b1c6; font-size:.85rem;}
.pill {display:inline-block; padding:5px 12px; border-radius:999px; font-size:.8rem; font-weight:700; color:#0b1020;}
.bar {height:8px; background:rgba(255,255,255,.10); border-radius:999px; overflow:hidden; margin:10px 0 12px;}
.bar > div {height:100%; border-radius:999px; width:var(--w); animation:grow 1s ease both;}
.stats {display:grid; grid-template-columns:1fr 1fr; gap:10px 14px; font-size:.92rem; margin-top:12px;}
.stats b {font-size:1.05rem; display:block;}
.chip {display:inline-block; background:rgba(255,255,255,.10); border-radius:8px; padding:3px 9px; margin:4px 4px 0 0; font-size:.8rem;}
.action {margin-top:14px; padding:10px 12px; border-radius:12px; font-size:.88rem; background:rgba(255,255,255,.06);}
.rank {width:36px; height:36px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  font-weight:800; color:#0b1020; background:linear-gradient(135deg,#00e0c6,#7c5cff); margin-bottom:12px;}
.pname {font-weight:700; font-size:1.02rem;}
.price {margin-top:8px; font-weight:800; color:#00e0c6;}
.section-title {font-size:1.25rem; font-weight:800; margin:22px 0 10px;}
.tbl {width:100%; border-collapse:collapse; background:rgba(255,255,255,.05); border-radius:14px; overflow:hidden;}
.tbl th {text-align:left; padding:12px 14px; background:rgba(124,92,255,.22); color:#fff; font-size:.85rem;}
.tbl td {padding:11px 14px; border-top:1px solid rgba(255,255,255,.08); color:#e6ebff; font-size:.92rem;}
.tbl tr:hover td {background:rgba(255,255,255,.04);}
.stTabs [data-baseweb="tab-list"] {gap:6px; background:rgba(255,255,255,.05); padding:6px; border-radius:14px;}
.stTabs [data-baseweb="tab"] {border-radius:10px; padding:9px 18px; font-weight:700; color:#a9b1c6;}
.stTabs [aria-selected="true"] {background:linear-gradient(135deg,#7c5cff,#3a7bd5) !important; color:#fff !important;}
.stTabs [data-baseweb="tab-highlight"] {display:none;}

/* Text input and select boxes */
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="select"] > div {
  background-color:#141a33 !important; border:1px solid rgba(255,255,255,.18) !important; border-radius:12px !important;}
div[data-baseweb="input"] input, div[data-baseweb="base-input"] input, textarea {
  background-color:#141a33 !important; color:#eef2ff !important; -webkit-text-fill-color:#eef2ff !important;}
input::placeholder, textarea::placeholder {color:#8b93a7 !important; -webkit-text-fill-color:#8b93a7 !important; opacity:1 !important;}
div[data-baseweb="select"] span, div[data-baseweb="select"] div {color:#eef2ff !important;}
div[data-baseweb="tag"] {background-color:#7c5cff !important; border-radius:8px !important;}
div[data-baseweb="tag"] span {color:#ffffff !important; -webkit-text-fill-color:#ffffff !important;}

/* Dropdown list (opens in a popup outside the app area) */
div[data-baseweb="popover"] > div, div[data-baseweb="popover"] ul, div[data-baseweb="menu"],
ul[role="listbox"], div[role="listbox"] {
  background-color:#141a33 !important; border:1px solid rgba(255,255,255,.18) !important; border-radius:12px !important;}
li[role="option"], ul[role="listbox"] li, div[data-baseweb="menu"] li {
  background-color:#141a33 !important; color:#eef2ff !important; -webkit-text-fill-color:#eef2ff !important;}
li[role="option"] *, ul[role="listbox"] li *, div[data-baseweb="menu"] li * {
  color:#eef2ff !important; -webkit-text-fill-color:#eef2ff !important; background-color:transparent !important;}
li[role="option"]:hover, li[role="option"][aria-selected="true"] {background-color:rgba(124,92,255,.40) !important;}

[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label, label {color:#d7dcf0 !important;}
[data-testid="stNumberInput"] button {background-color:#1c2447 !important; color:#eef2ff !important; border:none !important;}
.stButton > button, .stFormSubmitButton > button {border-radius:12px; border:none; font-weight:700;
  background:linear-gradient(135deg,#00e0c6,#7c5cff); color:#fff; padding:.55rem 1.1rem;}
.stButton > button:hover, .stFormSubmitButton > button:hover {transform:translateY(-2px); box-shadow:0 8px 22px rgba(124,92,255,.45);}
[data-testid="stForm"] {background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.12); border-radius:18px;}
[data-testid="stAlert"] {background:rgba(255,255,255,.07) !important; border-radius:14px;}
@media(max-width:900px) {.kpis,.grid3 {grid-template-columns:1fr;}}
</style>
""".replace("__BG__", background_uri())

def html_block(value):
    return "".join(line.strip() for line in str(value).splitlines() if line.strip())


def md(value):
    st.markdown(html_block(value), unsafe_allow_html=True)


def esc(value):
    return html.escape(str(value))


def safe_float(value, default=0.0):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def metric_pct(value):
    return f"{safe_float(value) * 100:.2f}%"


def table_html(headers, rows):
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


@st.cache_data(show_spinner="Loading ShopSegments data...")
def load_data():
    missing = [str(p.relative_to(BASE_DIR)) for p in REQUIRED if not p.exists()]
    if missing:
        return {"missing": missing}

    rfm = pd.read_pickle(PROCESSED_DIR / "rfm_clustered.pkl")
    if "customer_unique_id" in rfm.columns:
        rfm = rfm.set_index("customer_unique_id")
    rfm.index = rfm.index.astype(str)
    rfm.index.name = "customer_unique_id"
    needed = ["segment", "recency_days", "frequency", "monetary",
              "avg_order_value", "unique_products", "kmeans_cluster"]
    customers = rfm[needed].copy()

    products = pd.read_parquet(RESULTS_DIR / "products.parquet")
    products["product_id"] = products["product_id"].astype(str)
    products = products.drop_duplicates("product_id").set_index("product_id")

    cards = pd.DataFrame(load_json(RESULTS_DIR / "segment_cards.json"))

    recs = pd.read_parquet(RESULTS_DIR / "customer_recommendations.parquet")
    recs["customer_unique_id"] = recs["customer_unique_id"].astype(str)
    recs["product_id"] = recs["product_id"].astype(str)

    similar = pd.read_parquet(RESULTS_DIR / "item_similar_products.parquet")
    similar["product_id"] = similar["product_id"].astype(str)
    similar["similar_product_id"] = similar["similar_product_id"].astype(str)

    points = None
    points_path = PROCESSED_DIR / "umap_sample.pkl"
    if points_path.exists():
        candidate = pd.read_pickle(points_path).reset_index()
        if {"umap_x", "umap_y", "segment"}.issubset(candidate.columns):
            points = candidate

    return {
        "missing": [],
        "customers": customers,
        "products": products,
        "recs": recs,
        "similar": similar,
        "cards": cards,
        "metrics": load_json(RESULTS_DIR / "recommender_metrics.json"),
        "clustering": load_json(RESULTS_DIR / "clustering_comparison.json"),
        "k_table": pd.read_csv(RESULTS_DIR / "k_selection.csv"),
        "split": load_json(RESULTS_DIR / "split_info.json"),
        "points": points,
    }


@st.cache_resource(show_spinner="Loading recommendation models...")
def load_models():
    clustering = load(MODELS_DIR / "clustering_models.joblib")
    similarity = load_npz(MODELS_DIR / "item_similarity.npz").tocsr()
    items = pd.read_parquet(MODELS_DIR / "catalog_items.parquet")["product_id"].astype(str).tolist()
    config = load_json(MODELS_DIR / "scoring_config.json")
    if similarity.shape[0] != len(items):
        raise ValueError("Item similarity matrix and catalog size do not match.")
    return clustering, similarity, items, config


def category_of(pid, data):
    products = data["products"]
    pid = str(pid)
    return str(products.loc[pid, "category"]) if pid in products.index else "Unknown"


def short_id(pid):
    return str(pid)[:8]


def price_of(pid, data):
    products = data["products"]
    pid = str(pid)
    if pid in products.index:
        value = products.loc[pid, "price"]
        if pd.notna(value):
            return f"R$ {float(value):,.2f}"
    return "Price n/a"


def product_label(pid, data):
    return f"{category_of(pid, data)} · {short_id(pid)}"


def segment_from_rules(recency, frequency, monetary, cfg):
    r = 5 - int(np.searchsorted(cfg["recency_edges"], recency, side="right"))
    m = int(np.searchsorted(cfg["monetary_edges"], monetary, side="right")) + 1
    f = 5 if frequency >= 6 else 4 if frequency >= 4 else 3 if frequency == 3 else 2 if frequency == 2 else 1
    if f == 1 and r >= 4:
        return "New"
    if r >= 4 and f >= 2 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 2:
        return "Loyal"
    if r <= 2 and f >= 2:
        return "At Risk"
    if r <= 2 and f == 1:
        return "Lost"
    return "Potential Loyalist"


def recommend_for_products(selected, similarity, items, data, n=10):
    products = data["products"]
    selected = [str(p) for p in selected]
    picks, counts = [], {}

    def add(pid, respect_cap=True):
        if pid in selected or pid in picks:
            return False
        cat = category_of(pid, data)
        if respect_cap and counts.get(cat, 0) >= MAX_PER_CATEGORY:
            return False
        picks.append(pid)
        counts[cat] = counts.get(cat, 0) + 1
        return True

    item_index = {pid: i for i, pid in enumerate(items)}
    rows = [item_index[p] for p in selected if p in item_index]
    source = "Bought together"

    if rows:
        scores = np.asarray(similarity[rows].sum(axis=0)).ravel()
        scores[rows] = 0.0
        for idx in np.argsort(-scores, kind="stable"):
            if scores[idx] <= 0 or len(picks) >= n:
                break
            add(items[idx])

    if len(picks) < n:
        source = "Mixed" if picks else "Most popular"
        ranked = products.sort_values("orders", ascending=False)
        selected_cats = {category_of(p, data) for p in selected}
        same_cat = ranked[ranked["category"].astype(str).isin(selected_cats)].index
        for pid in list(same_cat) + list(ranked.index):
            if len(picks) >= n:
                break
            add(pid)

    if len(picks) < n:
        for pid in products.sort_values("orders", ascending=False).index:
            if len(picks) >= n:
                break
            add(pid, respect_cap=False)

    return picks[:n], source


def render_brand():
    md('<div class="brand"><div class="logo">🛒</div><div class="name">ShopSegments</div></div>')


def render_kpis(data):
    cards = data["cards"]
    total = int(pd.to_numeric(cards["customers"], errors="coerce").sum())
    metrics = data["metrics"]
    chosen = metrics.get("chosen_method", "item_cf")
    precision = safe_float(metrics["test_all_users"].get(chosen, {}).get("precision_at_10")) * 100
    md(f"""
    <div class="kpis">
      <div class="glass kpi"><div class="label">Segmented customers</div><div class="value">{total:,}</div></div>
      <div class="glass kpi"><div class="label">Customer groups</div><div class="value">{len(cards)}</div></div>
      <div class="glass kpi"><div class="label">Precision at 10</div><div class="value">{precision:.2f}%</div></div>
    </div>""")


def render_segments(data):
    cards = data["cards"].set_index("segment").reindex(SEGMENT_ORDER).dropna(how="all")
    max_share = max(safe_float(cards["customer_share"].max()), 1.0)
    parts = []
    for i, (segment, row) in enumerate(cards.iterrows()):
        color, icon = SEGMENT_STYLE.get(segment, ("#7c5cff", "●"))
        width = min(100, safe_float(row["customer_share"]) / max_share * 100)
        top_categories = row["top_categories"]
        if isinstance(top_categories, str):
            top_categories = [top_categories]
        chips = "".join(f'<span class="chip">{esc(c)}</span>' for c in top_categories)
        parts.append(f"""
        <div class="glass" style="animation-delay:{i * 0.07:.2f}s">
          <div class="accent" style="background:{color}"></div>
          <span class="pill" style="background:{color}">{icon} {esc(segment)}</span>
          <div class="big">{safe_int(row['customers']):,}</div>
          <div class="muted">{safe_float(row['customer_share']) * 100:.1f}% of customers</div>
          <div class="bar"><div style="--w:{width:.1f}%;background:{color}"></div></div>
          <div class="stats">
            <div>Avg spend<b>{safe_float(row['avg_spend']):.2f}</b></div>
            <div>Revenue<b>{safe_float(row['revenue_share']) * 100:.1f}%</b></div>
            <div>Days since order<b>{safe_float(row['avg_recency_days']):.0f}</b></div>
            <div>Avg orders<b>{safe_float(row['avg_orders']):.2f}</b></div>
          </div>
          <div style="margin-top:12px">{chips}</div>
          <div class="action"><b>Next step:</b> {esc(row['suggested_action'])}</div>
        </div>""")
    md(f'<div class="grid3">{"".join(parts)}</div>')

    points = data["points"]
    if points is None:
        st.info("Cluster map data is not available.")
        return
    points = points.copy()
    points["segment"] = pd.Categorical(points["segment"], categories=SEGMENT_ORDER, ordered=True)
    fig = px.scatter(
        points, x="umap_x", y="umap_y", color="segment",
        color_discrete_map={k: v[0] for k, v in SEGMENT_STYLE.items()},
        category_orders={"segment": SEGMENT_ORDER},
        hover_data={"recency_days": True, "frequency": True, "monetary": ":.2f", "umap_x": False, "umap_y": False},
        opacity=0.75, template="plotly_dark",
    )
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(height=520, legend_title_text="", xaxis_title="", yaxis_title="",
                      xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                      yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                      margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig)


def render_product_cards(picks, data, source):
    if not picks:
        st.info("No recommendations available.")
        return
    cards = "".join(
        f'<div class="glass" style="animation-delay:{i * 0.04:.2f}s">'
        f'<div class="rank">#{i + 1}</div>'
        f'<div class="pname">{esc(category_of(pid, data))}</div>'
        f'<div class="muted">Product {esc(short_id(pid))}</div>'
        f'<div class="price">{esc(price_of(pid, data))}</div></div>'
        for i, pid in enumerate(picks)
    )
    md(f'<div class="muted" style="margin:6px 0 10px">Source: {esc(source)}</div>'
       f'<div class="grid-prod">{cards}</div>')


def render_existing(data):
    customer_id = st.text_input("Customer ID", key="cid",
                                placeholder="Paste a customer ID and press Enter").strip()
    if not customer_id:
        md('<div class="muted">Paste a customer ID above.</div>')
        return
    customers = data["customers"]
    if customer_id not in customers.index:
        st.warning("This ID is not in the segmented customer dataset. Try another ID or use New customer.")
        return

    row = customers.loc[customer_id]
    segment = str(row["segment"])
    color, icon = SEGMENT_STYLE.get(segment, ("#7c5cff", "●"))
    left, right = st.columns([1, 2])
    with left:
        md(f"""
        <div class="glass">
          <div class="accent" style="background:{color}"></div>
          <span class="pill" style="background:{color}">{icon} {esc(segment)}</span>
          <div class="stats">
            <div>Orders<b>{safe_int(row['frequency'])}</b></div>
            <div>Total spend<b>{safe_float(row['monetary']):.2f}</b></div>
            <div>Avg order<b>{safe_float(row['avg_order_value']):.2f}</b></div>
            <div>Days since order<b>{safe_int(row['recency_days'])}</b></div>
          </div>
          <div class="action"><b>Next step:</b> {esc(SEGMENT_ACTIONS.get(segment, ''))}</div>
        </div>""")
    with right:
        recs = data["recs"]
        cust = recs[recs["customer_unique_id"] == customer_id].sort_values("rank").head(10)
        if cust.empty:
            st.info("No saved recommendations for this customer.")
            return
        render_product_cards(cust["product_id"].tolist(), data, "Saved customer recommendations")


def render_new_customer(data, similarity, items, cfg):
    options = sorted(items, key=lambda pid: product_label(pid, data))
    with st.form("new_customer"):
        c1, c2, c3 = st.columns(3)
        recency = c1.number_input("Days since last purchase", min_value=0, max_value=5000, value=30)
        frequency = c2.number_input("Number of orders", min_value=1, max_value=500, value=1)
        monetary = c3.number_input("Total amount spent", min_value=0.0, value=100.0, step=10.0)
        selected = st.multiselect("Products this customer bought (optional)", options,
                                  format_func=lambda pid: product_label(pid, data))
        submitted = st.form_submit_button("Predict segment and recommend")

    if not submitted:
        return

    segment = segment_from_rules(recency, frequency, monetary, cfg)
    aov = monetary / frequency
    picks, source = recommend_for_products(selected, similarity, items, data)
    color, icon = SEGMENT_STYLE[segment]

    left, right = st.columns([1, 2])
    with left:
        md(f"""
        <div class="glass">
          <div class="accent" style="background:{color}"></div>
          <span class="pill" style="background:{color}">{icon} {esc(segment)}</span>
          <div class="stats">
            <div>Orders<b>{int(frequency)}</b></div>
            <div>Total spend<b>{monetary:.2f}</b></div>
            <div>Avg order<b>{aov:.2f}</b></div>
            <div>Days since purchase<b>{int(recency)}</b></div>
          </div>
          <div class="action"><b>Next step:</b> {esc(SEGMENT_ACTIONS[segment])}</div>
        </div>""")
        st.caption("Segment is estimated using the saved RFM scoring rules.")
    with right:
        render_product_cards(picks, data, source)


def render_customer(data, similarity, items, cfg):
    mode = st.radio("Customer type", ["Existing customer", "New customer"], horizontal=True)
    if mode == "Existing customer":
        render_existing(data)
    else:
        render_new_customer(data, similarity, items, cfg)


def render_product(data):
    similar = data["similar"]
    product_ids = sorted(similar["product_id"].dropna().unique(), key=lambda pid: product_label(pid, data))
    if not product_ids:
        st.info("No similar product records available.")
        return
    pid = st.selectbox("Product", product_ids, format_func=lambda value: product_label(value, data))
    sim = similar[similar["product_id"] == pid].sort_values("rank")
    if sim.empty:
        st.info("Not enough purchase history for this product.")
        return

    max_sim = max(safe_float(sim["similarity"].max()), 1e-6)
    md(f"""
    <div class="glass" style="margin:10px 0 18px">
      <div class="accent" style="background:#7c5cff"></div>
      <div class="muted">Selected product</div>
      <div class="pname">{esc(category_of(pid, data))}</div>
      <div class="muted">Product {esc(short_id(pid))}</div>
    </div>""")

    cards = []
    for _, row in sim.iterrows():
        product_id = str(row["similar_product_id"])
        score = safe_float(row["similarity"])
        cards.append(f"""
        <div class="glass">
          <div class="rank">#{safe_int(row['rank'])}</div>
          <div class="pname">{esc(category_of(product_id, data))}</div>
          <div class="muted">Product {esc(short_id(product_id))}</div>
          <div class="price">{esc(price_of(product_id, data))}</div>
          <div class="bar"><div style="--w:{min(100, score / max_sim * 100):.1f}%;background:#00e0c6"></div></div>
          <div class="muted">Similarity: {score:.3f}</div>
        </div>""")
    md(f'<div class="grid-prod">{"".join(cards)}</div>')


def render_model(data):
    metrics = data["metrics"]
    chosen = metrics.get("chosen_method", "item_cf")
    test = metrics.get("test_all_users", {})
    validation = metrics.get("validation", {})
    colors = {"popularity": "#ffd166", "item_cf": "#9d8cff", "als": "#00e0c6"}

    md('<div class="section-title">Recommendation models (test period)</div>')
    cards = []
    for i, method in enumerate(["popularity", "item_cf", "als"]):
        result = test.get(method, {})
        precision = safe_float(result.get("precision_at_10")) * 100
        badge = ('<span class="pill" style="background:#00e0c6">Used for saved recommendations</span>'
                 if method == chosen else "")
        cards.append(f"""
        <div class="glass" style="animation-delay:{i * 0.08:.2f}s">
          {badge}
          <div class="pname" style="margin-top:10px">{esc(METHOD_LABEL[method])}</div>
          <div class="big">{precision:.3f}%</div>
          <div class="muted">Precision@10</div>
          <div class="bar"><div style="--w:{min(100, precision * 5):.1f}%;background:{colors[method]}"></div></div>
        </div>""")
    md(f'<div class="grid3">{"".join(cards)}</div>')

    rows = []
    for method in ["popularity", "item_cf", "als"]:
        r = test.get(method, {})
        rows.append([
            METHOD_LABEL[method],
            metric_pct(r.get("precision_at_10")),
            metric_pct(r.get("recall_at_10")),
            metric_pct(r.get("hit_rate")),
            metric_pct(r.get("coverage")),
        ])
    md(table_html(["Model", "Precision@10", "Recall@10", "Hit rate", "Catalogue coverage"], rows))
    st.caption(f"Evaluated on {safe_int(metrics.get('test_users_evaluated')):,} customers.")

    if isinstance(validation, dict) and validation:
        md('<div class="section-title">Validation results (used to choose the model)</div>')
        vrows = []
        for method in ["popularity", "item_cf", "als"]:
            r = validation.get(method, {})
            if isinstance(r, dict):
                vrows.append([
                    METHOD_LABEL[method],
                    metric_pct(r.get("precision_at_10")),
                    metric_pct(r.get("recall_at_10")),
                    metric_pct(r.get("coverage")),
                ])
        md(table_html(["Model", "Precision@10", "Recall@10", "Catalogue coverage"], vrows))

    clustering = data["clustering"]
    md('<div class="section-title">Customer segmentation (clustering)</div>')
    items = [
        ("Chosen groups (k)", clustering.get("chosen_k", "N/A")),
        ("K-Means silhouette", f"{safe_float(clustering.get('kmeans_silhouette')):.3f}"),
        ("GMM silhouette", f"{safe_float(clustering.get('gmm_silhouette')):.3f}"),
        ("DBSCAN clusters", clustering.get("dbscan_clusters", "N/A")),
        ("DBSCAN noise share", f"{safe_float(clustering.get('dbscan_noise_share')) * 100:.1f}%"),
        ("Agreement with rule segments (ARI)", f"{safe_float(clustering.get('kmeans_ari_vs_rule_segments')):.3f}"),
    ]
    md('<div class="grid3">' + "".join(
        f'<div class="glass"><div class="muted">{esc(label)}</div>'
        f'<div class="big" style="font-size:1.6rem">{esc(value)}</div></div>'
        for label, value in items
    ) + "</div>")

    k_table = data["k_table"]
    if {"k", "silhouette", "inertia"}.issubset(k_table.columns):
        left, right = st.columns(2)
        with left:
            fig = px.line(k_table, x="k", y="silhouette", markers=True, template="plotly_dark", title="Silhouette score")
            fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig)
        with right:
            fig = px.line(k_table, x="k", y="inertia", markers=True, template="plotly_dark", title="Elbow method")
            fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig)

    st.caption("Saved customer recommendations use Item-CF with popularity fallback. "
               "ALS is evaluated as a benchmark and is not used for saved customer lists.")


def main():
    md(CSS)
    try:
        data = load_data()
    except Exception as error:
        st.error("ShopSegments could not load its data files.")
        st.exception(error)
        return

    if data.get("missing"):
        st.error("Required project files are missing:")
        for filename in data["missing"]:
            st.write(f"- {filename}")
        return

    try:
        _, similarity, items, config = load_models()
    except Exception as error:
        st.error("ShopSegments could not load its recommendation models.")
        st.exception(error)
        return

    render_brand()
    render_kpis(data)
    tabs = st.tabs(["Groups", "Customer", "Products", "Model"])
    with tabs[0]:
        render_segments(data)
    with tabs[1]:
        render_customer(data, similarity, items, config)
    with tabs[2]:
        render_product(data)
    with tabs[3]:
        render_model(data)


main()