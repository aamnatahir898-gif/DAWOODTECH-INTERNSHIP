from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="DemandCast | Demand & Inventory",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
Z80 = 1.2816

COLORS = {
    "teal": "#0F9D8A",
    "blue": "#2563EB",
    "purple": "#7657D5",
    "amber": "#C27A12",
    "red": "#C2414B",
    "muted": "#64748B",
}
CATEGORY_COLORS = {
    "FOODS": "#0F9D8A",
    "HOBBIES": "#7657D5",
    "HOUSEHOLD": "#C27A12",
}
CATEGORY_LABELS = {
    "FOODS": "Foods",
    "HOBBIES": "Hobbies",
    "HOUSEHOLD": "Household",
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');

    :root {
        --ink: #172B4D;
        --muted: #52647D;
        --line: #DCE5F0;
        --surface: #FFFFFF;
        --canvas: #F3F6FB;
        --navy: #142B4A;
        --blue: #2563EB;
        --teal: #0F9D8A;
        --violet: #7657D5;
        --amber: #C27A12;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }
    .stApp {
        background: linear-gradient(135deg, #F4F7FC 0%, #EAF1FA 48%, #F7F9FC 100%);
        color: var(--ink);
    }
    .block-container {
        max-width: 1480px;
        padding-top: 1.55rem;
        padding-bottom: 2.5rem;
    }
    [data-testid="stHeader"] { background: rgba(243,246,251,.92); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #142B4A 0%, #1C3B60 100%);
        border-right: 1px solid #294B70;
    }
    [data-testid="stSidebar"] * { color: #F7FAFF !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] input {
        background: #FFFFFF !important;
        color: #172B4D !important;
        border-color: #C7D5E5 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] input { color: #172B4D !important; }
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button {
        background: #EAF1FA !important;
        color: #172B4D !important;
    }
    h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] {
        color: var(--ink);
    }
    .hero {
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 190px;
        padding: 30px 36px;
        margin-bottom: 23px;
        border-radius: 22px;
        border: 1px solid #244B76;
        background: linear-gradient(112deg, #142B4A 0%, #1E4770 58%, #176D75 100%);
        box-shadow: 0 14px 35px rgba(25, 54, 89, .16);
    }
    .hero-copy { position: relative; z-index: 2; max-width: 70%; }
    .hero-kicker {
        color: #9DE9DE;
        font-size: .75rem;
        font-weight: 800;
        letter-spacing: .15em;
    }
    .hero-title {
        color: #FFFFFF;
        font-family: 'Manrope', sans-serif;
        font-size: 2.45rem;
        font-weight: 800;
        line-height: 1.15;
        margin: 8px 0;
    }
    .hero-subtitle {
        color: #E5EEF8;
        font-size: .98rem;
        line-height: 1.6;
        margin: 0;
    }
    .hero-art { width: 245px; max-width: 29%; height: auto; opacity: .96; }
    .metric-card {
        min-height: 135px;
        padding: 19px 20px;
        border: 1px solid #DCE5F0;
        border-radius: 17px;
        background: #FFFFFF;
        box-shadow: 0 5px 18px rgba(31, 55, 85, .055);
    }
    .metric-label {
        color: #52647D;
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .055em;
        text-transform: uppercase;
    }
    .metric-value {
        color: #172B4D;
        font-size: 1.7rem;
        font-weight: 800;
        line-height: 1.25;
        margin: 9px 0 5px;
        overflow-wrap: anywhere;
    }
    .metric-help { color: #52647D; font-size: .82rem; }
    .status {
        padding: 15px 18px;
        margin: 17px 0 22px;
        border: 1px solid var(--status-color);
        border-left: 5px solid var(--status-color);
        border-radius: 13px;
        background: #FFFFFF;
        color: #263B57;
        box-shadow: 0 4px 14px rgba(31,55,85,.04);
        line-height: 1.55;
    }
    .status strong { color: var(--status-color); }
    .status-detail { color: #52647D; }
    div[data-testid="stTabs"] {
        background: #FFFFFF;
        padding: 8px 10px 0;
        border: 1px solid #DCE5F0;
        border-radius: 15px 15px 0 0;
    }
    div[data-testid="stTabs"] button {
        color: #52647D !important;
        font-weight: 700;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #174EA6 !important;
        border-bottom-color: #2563EB !important;
    }
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCE5F0;
        padding: 13px 16px;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(31,55,85,.04);
    }
    [data-testid="stMetricLabel"] p { color: #52647D !important; font-weight: 700; }
    [data-testid="stMetricValue"] { color: #172B4D !important; }
    [data-testid="stDataFrame"] {
        border: 1px solid #DCE5F0;
        border-radius: 12px;
        overflow: hidden;
        background: #FFFFFF;
    }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid #B8C9DD;
        background: #FFFFFF;
        color: #174EA6;
        font-weight: 700;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        border-color: #2563EB;
        color: #174EA6;
        background: #EEF5FF;
    }
    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #DCE5F0;
        border-radius: 12px;
    }
    .stCaption, [data-testid="stCaptionContainer"] { color: #52647D !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label, value, help_text, accent):
    return f"""
    <div class="metric-card" style="border-top:3px solid {accent}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-help">{help_text}</div>
    </div>
    """


def chart_layout(fig, height=430):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="DM Sans, sans-serif", color="#263B57", size=12),
        height=height,
        margin=dict(l=12, r=12, t=48, b=12),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0, font=dict(color="#263B57")),
    )
    fig.update_xaxes(gridcolor="#E8EEF5", zerolinecolor="#DCE5F0", tickfont=dict(color="#52647D"))
    fig.update_yaxes(gridcolor="#E8EEF5", zerolinecolor="#DCE5F0", tickfont=dict(color="#52647D"))
    return fig


def show_chart(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


@st.cache_data(show_spinner="Loading DemandCast data...")
def load_data():
    history = pd.read_csv(DATA_DIR / "history.csv", parse_dates=["date"])
    forecasts = pd.read_csv(DATA_DIR / "forecasts.csv", parse_dates=["date"])
    meta = pd.read_csv(DATA_DIR / "series_meta.csv")
    comparison = pd.read_csv(DATA_DIR / "model_comparison.csv")
    category_error = pd.read_csv(DATA_DIR / "error_by_category.csv")

    store_error_path = DATA_DIR / "error_by_store.csv"
    store_error = (
        pd.read_csv(store_error_path)
        if store_error_path.exists()
        else pd.DataFrame()
    )
    return history, forecasts, meta, comparison, category_error, store_error


required_files = [
    "history.csv",
    "forecasts.csv",
    "series_meta.csv",
    "model_comparison.csv",
    "error_by_category.csv",
]
missing_files = [name for name in required_files if not (DATA_DIR / name).is_file()]

if missing_files:
    st.error("Required data files are missing: " + ", ".join(missing_files))
    st.info("Place the notebook-generated CSV files inside the app/data/ folder.")
    st.stop()

try:
    history, forecasts, meta, comparison, category_error, store_error = load_data()
except Exception as exc:
    st.error(f"Could not load the DemandCast data: {exc}")
    st.stop()

required_columns = {
    "history": (history, {"id", "date", "sales"}),
    "forecasts": (forecasts, {"model", "id", "date", "yhat", "lo", "hi"}),
    "metadata": (meta, {"id", "item_id", "cat_id", "store_id"}),
    "comparison": (comparison, {"model", "WAPE(%)"}),
    "category error": (category_error, {"model"}),
}
for label, (frame, columns) in required_columns.items():
    absent = columns - set(frame.columns)
    if absent:
        st.error(f"{label} file is missing columns: {', '.join(sorted(absent))}")
        st.stop()

if history.empty or forecasts.empty or meta.empty or comparison.empty:
    st.error("One or more required CSV files contain no records. Run the notebook first.")
    st.stop()

if forecasts[["yhat", "lo", "hi"]].isna().any().any():
    st.error("Forecast data contains missing prediction values. Regenerate the notebook outputs.")
    st.stop()

if (forecasts["lo"] > forecasts["yhat"]).any() or (forecasts["hi"] < forecasts["yhat"]).any():
    st.error("Forecast interval values are inconsistent. Regenerate the notebook outputs.")
    st.stop()

if meta["id"].duplicated().any():
    st.error("series_meta.csv contains duplicate series IDs.")
    st.stop()

available_models = set(forecasts["model"].dropna().unique())
comparison = comparison[comparison["model"].isin(available_models)].copy()
if comparison.empty:
    st.error("No model in model_comparison.csv has a matching final forecast.")
    st.stop()

comparison = comparison.sort_values("WAPE(%)", ascending=True, na_position="last")
model_options = comparison["model"].tolist()
best_model = model_options[0]

meta = meta.copy()
meta["item_label"] = meta["item_id"].astype(str) + "  ·  " + meta["cat_id"].astype(str)

# Controls
with st.sidebar:
    st.markdown("### DemandCast")
    st.caption("M5 Walmart demand forecasting")

    stores = sorted(meta["store_id"].dropna().unique().tolist())
    selected_store = st.selectbox("Store", stores)

    store_meta = meta[meta["store_id"] == selected_store]
    categories = sorted(store_meta["cat_id"].dropna().unique().tolist())
    selected_category = st.selectbox(
        "Product category",
        categories,
        format_func=lambda value: CATEGORY_LABELS.get(value, value),
    )

    item_pool = store_meta[store_meta["cat_id"] == selected_category].sort_values("item_id")
    selected_item_label = st.selectbox("Item", item_pool["item_label"].tolist())
    selected_item_id = item_pool.loc[
        item_pool["item_label"] == selected_item_label, "id"
    ].iloc[0]

    selected_model = st.selectbox(
        "Forecast model",
        model_options,
        index=0,
        help="Models are ordered by overall walk-forward WAPE; lower is better.",
    )

    current_stock = st.number_input(
        "Current stock (units)",
        min_value=0,
        value=0,
        step=1,
    )

    st.divider()
    st.caption("Forecast horizon: 28 days")
    st.caption(f"Selected model: {selected_model}")

selected_history = (
    history[history["id"] == selected_item_id]
    .sort_values("date")
    .tail(120)
)
selected_forecast = (
    forecasts[
        (forecasts["id"] == selected_item_id)
        & (forecasts["model"] == selected_model)
    ]
    .sort_values("date")
    .copy()
)

if selected_forecast.empty:
    st.warning("No forecast is available for this item and model.")
    st.stop()

horizon_days = len(selected_forecast)
if horizon_days == 0:
    st.warning("The selected forecast has no dates.")
    st.stop()

# Inventory calculations use the same approximation as the notebook.
expected_demand = float(selected_forecast["yhat"].sum())
daily_sigma = np.maximum(
    (selected_forecast["hi"].to_numpy() - selected_forecast["lo"].to_numpy())
    / (2 * Z80),
    0,
)
aggregate_sigma = float(np.sqrt(np.square(daily_sigma).sum()))
aggregate_low = max(0.0, expected_demand - Z80 * aggregate_sigma)
aggregate_high = expected_demand + Z80 * aggregate_sigma
safety_stock = Z80 * aggregate_sigma
suggested_order = max(0.0, expected_demand + safety_stock - current_stock)

average_daily_demand = expected_demand / horizon_days
stock_cover = (
    current_stock / average_daily_demand
    if average_daily_demand > 0
    else np.inf
)

if current_stock < aggregate_low:
    stock_status = "Low stock against forecast demand"
    status_color = COLORS["red"]
    status_detail = "Available stock is below the approximate lower demand estimate."
elif current_stock > aggregate_high:
    stock_status = "Stock exceeds forecast demand range"
    status_color = COLORS["amber"]
    status_detail = "Available stock is above the approximate upper demand estimate."
else:
    stock_status = "Stock is within forecast demand range"
    status_color = COLORS["teal"]
    status_detail = "Available stock is within the approximate demand range."

cover_label = f"{stock_cover:.1f} days" if np.isfinite(stock_cover) else "Not applicable"
item_name = str(item_pool.loc[item_pool["id"] == selected_item_id, "item_id"].iloc[0])
category_name = CATEGORY_LABELS.get(selected_category, selected_category)
forecast_start = selected_forecast["date"].min().strftime("%d %b %Y")
forecast_end = selected_forecast["date"].max().strftime("%d %b %Y")

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-copy">
            <div class="hero-kicker">M5 WALMART DATASET · DEMAND INTELLIGENCE</div>
            <div class="hero-title">DemandCast</div>
            <p class="hero-subtitle">
                Forecast demand. Understand performance. Plan inventory.<br>
                {selected_store} &nbsp;·&nbsp; {category_name} &nbsp;·&nbsp; {item_name}
            </p>
        </div>
        <svg class="hero-art" viewBox="0 0 260 170" role="img" aria-label="Demand analytics illustration">
            <rect x="12" y="12" width="236" height="146" rx="17" fill="#FFFFFF" fill-opacity=".10" stroke="#FFFFFF" stroke-opacity=".28"/>
            <path d="M36 126 L36 102 L70 102 L70 82 L104 82 L104 91 L138 91 L138 58 L172 58 L172 69 L206 69 L206 40" fill="none" stroke="#9DE9DE" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M36 126 L70 102 L104 82 L138 91 L172 58 L206 40 L206 126 Z" fill="#9DE9DE" fill-opacity=".13"/>
            <circle cx="206" cy="40" r="7" fill="#F8C96A" stroke="#FFFFFF" stroke-width="3"/>
            <path d="M34 137 H222" stroke="#FFFFFF" stroke-opacity=".45" stroke-width="2"/>
            <rect x="43" y="23" width="43" height="8" rx="4" fill="#FFFFFF" fill-opacity=".65"/>
            <rect x="93" y="23" width="25" height="8" rx="4" fill="#9DE9DE"/>
        </svg>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_columns = st.columns(4)
kpis = [
    ("Expected demand", f"{expected_demand:,.0f}", f"Next {horizon_days} days", COLORS["blue"]),
    ("Approx. 80% range", f"{aggregate_low:,.0f} – {aggregate_high:,.0f}", "Total demand estimate", COLORS["purple"]),
    ("Safety stock", f"{safety_stock:,.0f}", "Approximate uncertainty buffer", COLORS["amber"]),
    ("Suggested order", f"{suggested_order:,.0f}", "After subtracting current stock", COLORS["teal"]),
]
for column, (label, value, helper, accent) in zip(kpi_columns, kpis):
    with column:
        st.markdown(metric_card(label, value, helper, accent), unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="status" style="--status-color:{status_color}">
        <strong>{stock_status}</strong> &nbsp; {status_detail}
        <span class="status-detail"> &nbsp; · &nbsp; Estimated stock cover: {cover_label}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

forecast_tab, store_tab, model_tab, category_tab = st.tabs(
    ["Forecast & inventory", "Store overview", "Model performance", "Error analysis"]
)

with forecast_tab:
    st.caption(f"{forecast_start} — {forecast_end} · {selected_model}")

    fig = go.Figure()
    if not selected_history.empty:
        fig.add_trace(
            go.Scatter(
                x=selected_history["date"],
                y=selected_history["sales"],
                name="Historical sales",
                mode="lines",
                line=dict(color=COLORS["muted"], width=2),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=selected_forecast["date"],
            y=selected_forecast["hi"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=selected_forecast["date"],
            y=selected_forecast["lo"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(45,212,191,.17)",
            name="Daily 80% prediction interval",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=selected_forecast["date"],
            y=selected_forecast["yhat"],
            name="Forecast",
            mode="lines+markers",
            line=dict(color=COLORS["teal"], width=3),
            marker=dict(size=6),
        )
    )
    fig.add_vline(
        x=pd.Timestamp(selected_forecast["date"].min()).to_pydatetime(),
        line_dash="dot",
        line_color="#94A3B8",
    )
    chart_layout(fig, 440)
    show_chart(fig)

    left, right = st.columns([3, 1])
    with left:
        st.markdown("**Daily forecast**")
        table = selected_forecast[["date", "yhat", "lo", "hi"]].copy()
        table.columns = ["Date", "Forecast", "Lower interval", "Upper interval"]
        table[["Forecast", "Lower interval", "Upper interval"]] = table[
            ["Forecast", "Lower interval", "Upper interval"]
        ].round(2)
        st.dataframe(table, use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Export**")
        st.download_button(
            "Download forecast CSV",
            data=selected_forecast.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_item_id}_{selected_model.replace(' ', '_')}_forecast.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download inventory summary",
            data=pd.DataFrame(
                [{
                    "store": selected_store,
                    "item_id": item_name,
                    "category": selected_category,
                    "model": selected_model,
                    "forecast_days": horizon_days,
                    "expected_demand": expected_demand,
                    "approx_80pct_lower": aggregate_low,
                    "approx_80pct_upper": aggregate_high,
                    "current_stock": current_stock,
                    "approx_safety_stock": safety_stock,
                    "suggested_order": suggested_order,
                }]
            ).to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_item_id}_inventory_plan.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.caption(
        "The daily shaded interval comes from the model's prediction bounds. "
        "The aggregate demand range and safety stock are approximations that assume "
        "independent daily forecast errors."
    )

with store_tab:
    st.markdown(f"**{selected_store} · forecast demand by item**")

    store_items = store_meta[["id", "item_id", "cat_id"]].drop_duplicates().copy()
    model_demand = (
        forecasts[forecasts["model"] == selected_model]
        .groupby("id")["yhat"]
        .sum()
    )
    store_items["expected_demand"] = store_items["id"].map(model_demand).fillna(0)
    category_totals = store_items.groupby("cat_id")["expected_demand"].sum()

    category_columns = st.columns(max(1, len(category_totals)))
    for column, (category, amount) in zip(category_columns, category_totals.items()):
        with column:
            st.metric(
                CATEGORY_LABELS.get(category, category),
                f"{amount:,.0f} units",
            )

    plot_data = store_items.sort_values("expected_demand", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=plot_data["expected_demand"],
            y=plot_data["item_id"],
            orientation="h",
            marker_color=[
                CATEGORY_COLORS.get(category, COLORS["blue"])
                for category in plot_data["cat_id"]
            ],
            customdata=plot_data["cat_id"].map(CATEGORY_LABELS),
            hovertemplate="%{y}<br>%{customdata}<br>%{x:,.1f} units<extra></extra>",
        )
    )
    chart_layout(fig, max(340, 38 * len(plot_data) + 100))
    fig.update_layout(hovermode="closest", showlegend=False)
    show_chart(fig)

    st.download_button(
        "Download store demand CSV",
        data=plot_data.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_store}_demand_by_item.csv",
        mime="text/csv",
    )

with model_tab:
    best_row = comparison.iloc[0]
    st.success(
        f"Best backtested model: {best_row['model']} · "
        f"WAPE {best_row['WAPE(%)']:.2f}% (lower is better)"
    )

    fig = go.Figure(
        go.Bar(
            x=comparison["model"],
            y=comparison["WAPE(%)"],
            marker_color=[
                COLORS["teal"] if name == best_model else "#526782"
                for name in comparison["model"]
            ],
            text=comparison["WAPE(%)"].round(2),
            textposition="outside",
        )
    )
    fig.update_layout(title="Walk-forward model comparison · WAPE (%)", showlegend=False)
    chart_layout(fig, 390)
    show_chart(fig)

    metric_aliases = {
        "MAE": "MAE",
        "WAPE(%)": "WAPE (%)",
        "sMAPE(%)": "sMAPE (%)",
        "MAPE(%)": "MAPE (%)",
        "Coverage 80% (%)": "80% interval coverage (%)",
        "MAPE_nonzero(%)": "MAPE, non-zero actuals (%)",
        "PI coverage(%)": "Prediction interval coverage (%)",
        "WAPE improvement vs Seasonal Naive (%)": "WAPE improvement vs Seasonal Naive (%)",
    }
    display_columns = [column for column in metric_aliases if column in comparison.columns]
    display_table = comparison[["model"] + display_columns].rename(
        columns={column: metric_aliases[column] for column in display_columns}
    )
    st.dataframe(
        display_table.round(2),
        use_container_width=True,
        hide_index=True,
    )

    coverage_column = next(
        (column for column in ["Coverage 80% (%)", "PI coverage(%)"] if column in comparison.columns),
        None,
    )
    if coverage_column:
        low_coverage = comparison.loc[
            comparison[coverage_column] < 75, "model"
        ].tolist()
        if low_coverage:
            st.warning(
                "Prediction interval coverage is below 75% for: "
                + ", ".join(low_coverage)
                + ". Interpret these uncertainty ranges cautiously."
            )

    st.download_button(
        "Download model comparison CSV",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

with category_tab:
    st.markdown("**Forecast error by product category · WAPE (%)**")
    category_columns = [column for column in category_error.columns if column != "model"]
    if category_columns:
        fig = go.Figure()
        for category in category_columns:
            fig.add_trace(
                go.Bar(
                    name=CATEGORY_LABELS.get(category, category),
                    x=category_error["model"],
                    y=category_error[category],
                    marker_color=CATEGORY_COLORS.get(category, COLORS["blue"]),
                )
            )
        fig.update_layout(barmode="group", title="Category-wise WAPE · lower is better")
        chart_layout(fig, 420)
        show_chart(fig)
        st.dataframe(category_error.round(2), use_container_width=True, hide_index=True)
    else:
        st.info("No category-wise error columns are available.")

    if not store_error.empty:
        st.markdown("**Forecast error by store · WAPE (%)**")
        st.dataframe(store_error.round(2), use_container_width=True, hide_index=True)
    else:
        st.caption(
            "Store-wise error CSV was not included in app/data. "
            "The notebook may still have saved it in the outputs folder."
        )

with st.expander("Method, inventory calculation and limitations"):
    st.markdown(
        """
        - **Forecast horizon:** 28 days, using the selected model's saved predictions.
        - **Validation:** model comparison uses chronological walk-forward backtesting,
          not a random train/test split.
        - **Baseline:** Seasonal Naive provides the reference WAPE for model improvement.
        - **Prediction interval:** the shaded daily band shows the saved lower and upper
          model bounds.
        - **Suggested order:** expected 28-day demand + approximate safety stock −
          current stock, with a minimum order quantity of zero.
        - **Uncertainty assumption:** aggregate uncertainty combines daily standard
          deviation estimates using an independence assumption. This is an approximation,
          not a calibrated guarantee of 80% coverage for the 28-day total.
        - **Operational limitation:** lead time, supplier constraints, minimum order
          quantities, storage capacity and item-to-item demand dependence are not modelled.
        - **Dataset limitation:** M5 represents Walmart stores in the United States.
          Ramadan and Eid indicators come from the M5 event calendar and should not be
          interpreted as Pakistani consumer behaviour.
        """
    )

st.caption(
    f"DemandCast · M5 Walmart dataset · {len(meta)} selected item-store series · "
    f"Forecast period: {forecast_start} to {forecast_end}"
)
