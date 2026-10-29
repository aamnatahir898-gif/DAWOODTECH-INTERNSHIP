# PriceSense — used-car price estimator
import base64, html
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PriceSense", page_icon="🚘", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "pricesense_final_pipeline.joblib"
YEAR_REF = 2022
HERO = "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=2200&q=85"
TEXT = ["title", "car_features"]
CAT = ["location", "engine_type", "transmission", "registered_in", "color", "assembly", "body_type"]
NUM = ["mileage_km", "engine_capacity_cc", "model_year_clean", "car_age", "mileage_per_year"]
FEATURES = TEXT + CAT + NUM
CITIES = ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad", "Peshawar", "Multan", "Quetta", "Gujranwala", "Hyderabad", "Sialkot", "Bahawalpur", "Sargodha", "Abbottabad"]

st.session_state.setdefault("page", "home")
st.session_state.setdefault("result", None)
def go(page): st.session_state.page = page

def esc(value): return html.escape(str(value))

CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root{--blue:#38bdf8;--violet:#818cf8;--muted:#9db2cb;--line:rgba(148,190,235,.18)}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}.stApp{color:#eaf2fc;background:radial-gradient(900px 500px at 95% 0%,#102a49 0%,transparent 60%),#050b18}
#MainMenu,footer{visibility:hidden}.block-container{max-width:1160px;padding-top:1.7rem;padding-bottom:3rem}
@keyframes rise{from{opacity:0;transform:translateY(17px)}to{opacity:1;transform:translateY(0)}}
@keyframes glow{0%,100%{box-shadow:0 12px 36px #0003}50%{box-shadow:0 12px 42px #38bdf833}}
/* Sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1c36,#050b18);border-right:1px solid var(--line)}
[data-testid="stSidebar"]>div:first-child{padding-top:1.2rem}
.brand{display:flex;align-items:center;gap:11px;font:800 1.45rem Manrope,sans-serif;color:white;letter-spacing:-.04em}
.brand-icon{display:grid;place-items:center;width:44px;height:44px;border-radius:15px;background:linear-gradient(135deg,var(--blue),var(--violet));box-shadow:0 8px 25px #38bdf844;font-size:1.4rem}
.brand-sub{color:var(--muted);font-size:.7rem;text-transform:uppercase;letter-spacing:.16em;margin:.45rem 0 1.5rem}
[data-testid="stSidebar"] .stButton button{width:100%;text-align:left;justify-content:flex-start;border-radius:14px;padding:.75rem .9rem;font-weight:700;transition:.2s}
[data-testid="stSidebar"] button[kind="secondary"]{background:#ffffff05;border:1px solid transparent;color:#d4e3f5}
[data-testid="stSidebar"] button[kind="secondary"]:hover{background:#38bdf81a;border-color:#38bdf855;transform:translateX(3px);color:white}
[data-testid="stSidebar"] button[kind="primary"]{background:linear-gradient(100deg,#38bdf82c,#818cf82c);border:1px solid #38bdf877;color:white}
.side-info{border:1px solid var(--line);background:#ffffff06;border-radius:15px;padding:13px;color:var(--muted);font-size:.8rem;line-height:1.6;margin-top:1.4rem}

.head{animation:rise .5s ease-out;margin-bottom:1.15rem}.title{font:800 clamp(1.8rem,3vw,2.5rem) Manrope,sans-serif;letter-spacing:-.04em;color:#fff}.subtitle{color:var(--muted);margin-top:.25rem}
.sec{font:800 1.05rem Manrope,sans-serif;color:#f1f7ff}.sec small{display:inline-grid;place-items:center;background:linear-gradient(100deg,var(--blue),var(--violet));color:#061120;border-radius:8px;padding:4px 8px;margin-right:8px;font-size:.72rem}
.sec-sub{color:var(--muted);font-size:.85rem;margin:.35rem 0 .8rem}
div[class*="st-key-panel_"]{background:linear-gradient(145deg,#102039eF,#091629f2);border:1px solid var(--line)!important;border-radius:21px;padding:.75rem .9rem .45rem;margin-bottom:1rem;animation:rise .5s ease-out both;box-shadow:0 12px 35px #0002}
[data-testid="stWidgetLabel"] p,label p{color:#dbe9f8!important;font-weight:600;font-size:.9rem}
.stTextInput input,.stTextArea textarea,.stNumberInput input{background:#fff!important;color:#0f172a!important;-webkit-text-fill-color:#0f172a!important;border:1px solid #cbd5e1!important;border-radius:12px!important}
.stTextInput input::placeholder,.stTextArea textarea::placeholder{color:#64748b!important;-webkit-text-fill-color:#64748b!important;opacity:1!important}
[data-baseweb="select"]>div{background:#fff!important;border:1px solid #cbd5e1!important;border-radius:12px!important;min-height:2.65rem}
[data-baseweb="select"] *{color:#0f172a!important;-webkit-text-fill-color:#0f172a!important}[data-baseweb="popover"],[data-baseweb="menu"]{background:#fff!important}[data-baseweb="popover"] li,[data-baseweb="popover"] li *{color:#0f172a!important}
[data-testid="stNumberInput"] button{background:#f1f5f9!important;color:#0f172a!important}
[data-testid="stMain"] .stButton>button[kind="primary"]{border:0;border-radius:14px;padding:.85rem 1rem;font-weight:800;color:#04111f;background:linear-gradient(100deg,#7dd3fc,#a5b4fc);box-shadow:0 10px 27px #38bdf833;transition:.2s}
[data-testid="stMain"] .stButton>button[kind="primary"]:hover{transform:translateY(-2px);filter:brightness(1.07);box-shadow:0 14px 35px #38bdf855}
[data-testid="stMain"] .stButton>button[kind="primary"] p{color:#04111f;font-weight:800}
/* Result and feature cards */
.result{position:relative;overflow:hidden;border:1px solid #38bdf866;border-radius:24px;padding:1.5rem 1.7rem;margin:1rem 0;background:linear-gradient(125deg,#123e64,#0a1c36);animation:rise .55s ease-out,glow 3s ease-in-out .6s 2}
.eyebrow{color:#b8d7ef;text-transform:uppercase;letter-spacing:.13em;font-size:.74rem;font-weight:800}.price{font:800 clamp(2.3rem,5vw,3.8rem) Manrope,sans-serif;letter-spacing:-.05em;color:white;margin:.2rem 0}.muted{color:var(--muted);font-size:.8rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:.8rem;margin:1rem 0}.mini{border:1px solid var(--line);background:linear-gradient(145deg,#102039,#091629);border-radius:17px;padding:1rem;animation:rise .45s ease-out both;transition:transform .2s,border-color .2s}.mini:hover{transform:translateY(-4px);border-color:#38bdf877}.mini-label{color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700}.mini-value{font-size:1.05rem;font-weight:800;color:#f1f7ff;margin-top:.3rem;overflow-wrap:anywhere}
.feature-card{border:1px solid var(--line);border-radius:20px;background:linear-gradient(145deg,#102039,#091629);padding:1.2rem 1.3rem;margin:.8rem 0;animation:rise .5s ease-out both}.fi-row{display:grid;grid-template-columns:minmax(100px,220px) 1fr 48px;gap:.7rem;align-items:center;margin:.75rem 0}.fi-name{font-size:.83rem;color:#e3eefb;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.track{height:10px;background:#ffffff12;border-radius:99px;overflow:hidden}.bar{height:100%;width:var(--w);background:linear-gradient(90deg,#38bdf8,#818cf8);border-radius:99px;animation:rise .8s ease-out}.pct{text-align:right;color:#bfe7fb;font-size:.8rem;font-weight:800}
.compare{border:1px solid var(--line);background:linear-gradient(145deg,#102039,#091629);border-radius:20px;padding:1.2rem 1.3rem;margin:1rem 0;animation:rise .5s ease-out}.chip{display:inline-block;border-radius:99px;padding:.45rem .8rem;border:1px solid #38bdf855;color:#bae6fd;background:#38bdf81a;font-size:.85rem;font-weight:700}
.note{color:#8aa0b9;font-size:.78rem;text-align:center;margin-top:1.3rem}
/* Welcome */
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="stHeader"]{ }
.welcome{min-height:64vh;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;border-radius:28px;overflow:hidden;padding:3rem 1rem;margin:1rem 0 1.7rem;background:linear-gradient(180deg,#03081233,#03081299),url('__HERO__') center 58%/cover no-repeat;box-shadow:0 20px 65px #0006;animation:rise .8s ease-out}
.wordmark{font:800 clamp(3.3rem,10vw,7.7rem) Manrope,sans-serif;letter-spacing:-.07em;color:white;text-shadow:0 8px 35px #0009;line-height:1.1}.wordmark span{background:linear-gradient(100deg,#7dd3fc,#a5b4fc);-webkit-background-clip:text;background-clip:text;color:transparent}.welcome-line{width:100px;height:4px;border-radius:99px;background:linear-gradient(90deg,#38bdf8,#818cf8);margin:1.2rem auto 0}.welcome-caption{font-size:.95rem;color:#edf7ff;letter-spacing:.12em;text-transform:uppercase;margin-top:1rem;text-shadow:0 2px 12px #000}
@media(max-width:700px){.block-container{padding-left:1rem;padding-right:1rem}.fi-row{grid-template-columns:100px 1fr 43px}.welcome{min-height:52vh}}
</style>
"""

@st.cache_resource(show_spinner=False)
def load_model(path): return joblib.load(path)

def pick(label, choices, key):
    """Keep the manual-entry box in the same section and retain its value on reruns."""
    options = ["Select…"] + choices + ["Other / type manually"]
    selected = st.selectbox(label, options, key=f"{key}_choice")
    if selected == "Other / type manually":
        return st.text_input(f"Type {label.lower()}", key=f"{key}_manual", placeholder=f"Enter {label.lower()} here").strip()
    return "" if selected == "Select…" else selected

def feature_frame(pipe, limit=10):
    try:
        model, prep = pipe.named_steps["model"], pipe.named_steps["preprocessor"]
        values, names = np.asarray(model.feature_importances_, float), prep.get_feature_names_out()
        if len(values) != len(names) or values.sum() <= 0: return None
        df = pd.DataFrame({"name": names, "value": values / values.sum()}).nlargest(limit, "value")
        def readable(n):
            for prefix, label in [("text_title__", "Title: "), ("text_car_features__", "Feature: "), ("numeric__", "")]:
                if n.startswith(prefix): return label + n[len(prefix):].replace("_", " ")
            if n.startswith("categorical__"):
                return n[len("categorical__"):].replace("_", " ").capitalize()
            return n.replace("_", " ").capitalize()
        df["label"] = df["name"].map(readable)
        return df.reset_index(drop=True)
    except Exception: return None

def show_features(pipe, limit=10):
    df = feature_frame(pipe, limit)
    if df is None or df.empty: return
    peak = df.value.max()
    rows = "".join(f'<div class="fi-row"><div class="fi-name" title="{esc(r.label)}">{esc(r.label[:30] + ("…" if len(r.label)>30 else ""))}</div><div class="track"><div class="bar" style="--w:{r.value/peak*100:.1f}%"></div></div><div class="pct">{r.value*100:.1f}%</div></div>' for r in df.itertuples())
    st.markdown(f'<div class="feature-card"><div class="sec">✨ What influences the model</div><div class="sec-sub">Overall feature importance across the trained model.</div>{rows}</div>', unsafe_allow_html=True)
    with st.expander("View full feature names"):
        st.dataframe(df[["name", "value"]].rename(columns={"name":"Feature", "value":"Importance"}), hide_index=True, use_container_width=True)

def card_grid(items):
    cards = "".join(f'<div class="mini" style="animation-delay:{i*.06:.2f}s"><div class="mini-label">{esc(k)}</div><div class="mini-value">{esc(v)}</div></div>' for i,(k,v) in enumerate(items))
    st.markdown(f'<div class="cards">{cards}</div>', unsafe_allow_html=True)

def header(title, subtitle):
    st.markdown(f'<div class="head"><div class="title">{esc(title)}</div><div class="subtitle">{esc(subtitle)}</div></div>', unsafe_allow_html=True)

def section(num, title, subtitle):
    st.markdown(f'<div class="sec"><small>{num}</small>{esc(title)}</div><div class="sec-sub">{esc(subtitle)}</div>', unsafe_allow_html=True)

def welcome():
    st.markdown(f'<div class="welcome" style="background-image:linear-gradient(180deg,#03081228,#030812a8),url(\'{HERO}\')"><div class="wordmark">Price<span>Sense</span></div><div class="welcome-line"></div><div class="welcome-caption">Smarter estimates. Better decisions.</div></div>', unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid: st.button("Continue to PriceSense  →", type="primary", use_container_width=True, on_click=go, args=("estimate",))

def sidebar():
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-icon">🚘</div>PriceSense</div><div class="brand-sub">Used car price studio</div>', unsafe_allow_html=True)
        pages = [("estimate","🚘  Estimate price"),("insights","✦  Model insights"),("about","ⓘ  About PriceSense"),("home","⌂  Home")]
        for key,label in pages:
            st.button(label, key=f"nav_{key}", type="primary" if st.session_state.page==key else "secondary", on_click=go, args=(key,))
        st.markdown('<div class="side-info"><b>Quick guide</b><br>01 &nbsp; Enter car details<br>02 &nbsp; Get an estimated price<br>03 &nbsp; Compare a seller’s asking price<br><br>Estimates are indicative, not guaranteed.</div>', unsafe_allow_html=True)

def estimate(pipe):
    header("Estimate your car’s price", "Enter the details below to get an estimated price in Pakistani rupees.")
    with st.container(border=True, key="panel_vehicle"):
        section("01", "Vehicle details", "Choose an option or select “Other / type manually” to enter your own value here.")
        title = st.text_input("Car make, model and variant", placeholder="e.g. Toyota Corolla Altis 1.6", key="car_title")
        c1,c2,c3 = st.columns(3)
        with c1:
            location = pick("Current location", CITIES, "location")
            engine = pick("Engine / fuel type", ["Petrol","Diesel","Hybrid","CNG","LPG","Electric"], "engine")
            trans = pick("Transmission", ["Manual","Automatic"], "transmission")
        with c2:
            registered = pick("Registered in", CITIES + ["Unregistered"], "registered")
            color = pick("Exterior color", ["White","Black","Silver","Grey","Blue","Red","Brown","Maroon","Green","Golden","Beige","Other"], "color")
        with c3:
            assembly = pick("Assembly", ["Local","Imported"], "assembly")
            body = pick("Body type", ["Sedan","Hatchback","SUV","Crossover","MPV","Van","Double Cabin","Pickup","Coupe","Convertible","Truck"], "body")
        features = st.text_area("Car features (optional)", placeholder="e.g. sunroof, alloy rims, airbags, reverse camera", height=85, key="features")
    with st.container(border=True, key="panel_specs"):
        section("02", "Specifications", "Use the values from the car listing or documents.")
        a,b,c = st.columns(3)
        with a: mileage = st.number_input("Mileage (km)", min_value=1, max_value=1_500_000, value=66000, step=1000)
        with b: cc = st.number_input("Engine capacity (cc)", min_value=500, max_value=6000, value=1300, step=50)
        with c: year = st.number_input("Model year", min_value=1990, max_value=YEAR_REF+1, value=2018, step=1)
    with st.container(border=True, key="panel_compare"):
        section("03", "Seller’s asking price", "Optional: compare the seller’s price with the model estimate.")
        compare = st.checkbox("I want to compare an asking price", key="compare_toggle")
        asking = st.number_input("Asking price (PKR)", min_value=1, value=3_000_000, step=50_000, key="asking_price") if compare else None
    if st.button("✦  Estimate car price", type="primary", use_container_width=True, key="estimate_submit"):
        payload = {"title":title.strip(), "car_features":features.strip(), "location":location, "engine_type":engine, "transmission":trans, "registered_in":registered, "color":color, "assembly":assembly, "body_type":body, "mileage_km":float(mileage), "engine_capacity_cc":float(cc), "model_year_clean":int(year), "car_age":max(YEAR_REF-int(year),0), "mileage_per_year":float(mileage)/max(YEAR_REF-int(year),1) if int(year)<YEAR_REF else 0.0}
        errors=[]
        if not payload["title"]: errors.append("Enter the car make, model and variant.")
        if not 1990 <= int(year) <= YEAR_REF+1: errors.append(f"Model year must be between 1990 and {YEAR_REF+1}, matching the current training setup.")
        for field in CAT:
            if not str(payload.get(field, "")).strip(): errors.append(f"Choose or manually enter {field.replace('_',' ')}.")
        if errors:
            st.session_state.result=None
            for error in errors: st.error(error)
        else:
            try:
                X = pd.DataFrame([payload], columns=FEATURES)
                for col in TEXT+CAT: X[col] = X[col].fillna("").astype(str)
                with st.spinner("Analysing vehicle details…"):
                    price = float(pipe.predict(X)[0])
                if not np.isfinite(price): raise ValueError("The model returned an invalid prediction.")
                price=max(price,0.0)
                st.session_state.result={"price":price,"title":title.strip(),"year":int(year),"mileage":mileage,"cc":cc,"trans":trans,"location":location,"asking":asking}
            except Exception as exc:
                st.session_state.result=None
                st.error("Prediction failed. Check the saved pipeline and its expected input columns.")
                st.code(f"{type(exc).__name__}: {exc}")
    result=st.session_state.result
    if result:
        price=result["price"]
        words=f"≈ {price/10_000_000:.2f} crore" if price>=10_000_000 else (f"≈ {price/100_000:.1f} lacs" if price>=100_000 else "")
        st.markdown(f'<div class="result"><div class="eyebrow">Estimated market price · PKR</div><div class="price">₨ {price:,.0f}</div><div style="color:#9fd8f5;font-weight:700">{words}</div><div style="display:flex;justify-content:space-between;margin-top:1rem;font-weight:700"><span>₨ {price*.9:,.0f}</span><span>₨ {price*1.1:,.0f}</span></div><div class="muted" style="margin-top:.4rem">Indicative ±10% range, not a statistical confidence interval.</div></div>', unsafe_allow_html=True)
        card_grid([("Vehicle",result["title"]),("Model year",result["year"]),("Mileage",f'{result["mileage"]:,.0f} km'),("Engine",f'{result["cc"]:,.0f} cc'),("Transmission",result["trans"]),("Location",result["location"])])
        asking=result["asking"]
        if asking:
            diff=asking-price; pct=(diff/price*100) if price else 0
            status="Seller is asking more than the estimate" if diff>0 else ("Seller is asking less than the estimate" if diff<0 else "Asking price matches the estimate")
            st.markdown(f'<div class="compare"><div class="sec">💰 Asking price comparison</div><div class="cards"><div class="mini"><div class="mini-label">Seller asking</div><div class="mini-value">₨ {asking:,.0f}</div></div><div class="mini"><div class="mini-label">Model estimate</div><div class="mini-value">₨ {price:,.0f}</div></div><div class="mini"><div class="mini-label">Difference</div><div class="mini-value">₨ {abs(diff):,.0f}</div></div></div><span class="chip">{esc(status)}{f" ({abs(pct):.1f}%)" if price else ""}</span></div>', unsafe_allow_html=True)
        show_features(pipe)
    st.markdown('<div class="note">Predictions use historical listings. Condition, inspection and negotiation affect the actual market price.</div>', unsafe_allow_html=True)

def insights(pipe):
    header("Model insights", "A quick look at the trained model behind your estimate.")
    model=pipe.named_steps.get("model") if hasattr(pipe,"named_steps") else None
    card_grid([("Model",type(model).__name__ if model is not None else "Saved pipeline"),("Input features",len(FEATURES)),("Training workflow","Unchanged")])
    show_features(pipe,15)

def about():
    header("About PriceSense", "A simple used-car price estimator for Pakistan.")
    card_grid([("01 · Enter","Car details and specifications"),("02 · Estimate","Get a model-based price estimate"),("03 · Compare","Check the seller’s asking price")])
    st.markdown('<div class="feature-card">PriceSense provides an indicative estimate from a previously trained machine-learning pipeline. It does not guarantee the final selling price. Always consider vehicle condition, service history, inspection and current market demand.</div>',unsafe_allow_html=True)

# Apply the page styles
st.markdown(CSS, unsafe_allow_html=True)
page=st.session_state.page
if page == "home":
    st.markdown('<style>[data-testid="stSidebar"]{display:none!important}[data-testid="stSidebarCollapsedControl"]{display:none!important}</style>',unsafe_allow_html=True)
    welcome()
    st.stop()
sidebar()
if not MODEL.exists():
    st.error("Trained model not found. Keep pricesense_final_pipeline.joblib inside the models folder beside app.py.")
    st.code(str(MODEL)); st.stop()
try: pipe=load_model(str(MODEL))
except Exception as exc:
    st.error("Could not load the saved pipeline. Check installed dependencies and the scikit-learn version used when it was saved.")
    st.code(f"{type(exc).__name__}: {exc}"); st.stop()
if page=="insights": insights(pipe)
elif page=="about": about()
else: estimate(pipe)
