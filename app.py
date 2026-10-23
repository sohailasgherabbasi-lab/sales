import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from utils.data_loader import load_and_prepare
from utils.feature_engineering import add_features
from utils import analytics as an
from utils import modeling as md


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Sales Analytics & Profit Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# GLOBAL CSS — LIGHT THEME
# ============================================================
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', 'Inter', sans-serif; }
.stApp { background: linear-gradient(180deg, #F8F9FC 0%, #EEF1F8 100%); }
[data-testid="collapsedControl"] { display: none; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #6C5CE7 0%, #A29BFE 50%, #00CEC9 100%);
    padding: 3rem 2rem; border-radius: 22px; color: white;
    box-shadow: 0 12px 40px rgba(108,92,231,0.25);
    margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.hero::before {
    content: ""; position: absolute; top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.15), transparent);
    border-radius: 50%;
}
.hero h1 { font-size: 2.6rem; font-weight: 800; margin: 0 0 0.5rem 0; color: #FFFFFF !important; }
.hero p { font-size: 1.1rem; opacity: 0.95; margin: 0; }

/* Navbar */
div[data-testid="stHorizontalBlock"] > div > div > div > button {
    border-radius: 12px;
    border: 1px solid #E8ECF4;
    background: #FFFFFF;
    color: #4A5568;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.7rem 0.3rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    transition: all 0.25s ease;
    width: 100%;
}
div[data-testid="stHorizontalBlock"] > div > div > div > button:hover {
    background: #EEF1F8;
    border-color: #A29BFE;
    color: #6C5CE7;
    transform: translateY(-2px);
}
div[data-testid="stHorizontalBlock"] > div > div > div > button[kind="primary"] {
    background: linear-gradient(135deg, #6C5CE7, #A29BFE) !important;
    color: white !important;
    border-color: transparent !important;
    box-shadow: 0 6px 20px rgba(108,92,231,0.35) !important;
}

/* Cards */
.card {
    background: #FFFFFF; border: 1px solid #E8ECF4; padding: 1.5rem;
    border-radius: 18px; box-shadow: 0 6px 24px rgba(0,0,0,0.05);
    transition: all 0.3s ease; height: 100%;
}
.card:hover { transform: translateY(-4px); box-shadow: 0 12px 32px rgba(108,92,231,0.15); border-color: #A29BFE; }
.card h3 { color: #6C5CE7 !important; font-size: 1.15rem; margin-top: 0; }
.card p, .card ul { color: #4A5568; font-size: 0.95rem; }

/* KPI */
.kpi-card {
    background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FC 100%);
    border: 1px solid #E8ECF4; border-left: 5px solid #6C5CE7;
    padding: 1.2rem 1.4rem; border-radius: 14px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    transition: all 0.25s ease; height: 100%;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(108,92,231,0.18); }
.kpi-label { color: #8892A6; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
.kpi-value { color: #1A1D29; font-size: 1.8rem; font-weight: 800; margin-top: 0.3rem; }

/* Section title */
.section-title {
    color: #1A1D29; font-size: 1.5rem; font-weight: 700;
    margin: 2rem 0 1rem 0; display: flex; align-items: center; gap: 10px;
}
.section-title::before {
    content: ""; width: 6px; height: 28px;
    background: linear-gradient(180deg, #6C5CE7, #00CEC9); border-radius: 3px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6C5CE7, #A29BFE);
    color: white; border: none; border-radius: 12px;
    padding: 0.6rem 1.6rem; font-weight: 600;
    box-shadow: 0 4px 14px rgba(108,92,231,0.3);
    transition: all 0.25s ease;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 22px rgba(108,92,231,0.45); }

/* Tables */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; box-shadow: 0 4px 16px rgba(0,0,0,0.05); }

/* Alerts */
.stAlert { border-radius: 12px; border-left-width: 5px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# AUTO-CREATE DEMO FILE (only once)
# ============================================================
DEMO_FILE = Path("general_sales.csv")

if not DEMO_FILE.exists():
    rng = np.random.default_rng(0)
    n_demo = 1000
    demo_df = pd.DataFrame({
        "Order_ID":         [f"ORD-{i:04d}" for i in range(1, n_demo + 1)],
        "Order_Date":       pd.date_range("2023-01-01", periods=n_demo, freq="D"),
        "Customer_ID":      [f"C-{rng.integers(1, 100):03d}" for _ in range(n_demo)],
        "Customer_Name":    [f"Customer_{rng.integers(1, 100)}" for _ in range(n_demo)],
        "Product_Category": rng.choice(
            ["Electronics", "Furniture", "Office Supplies", "Clothing"], n_demo),
        "Sub_Category":     rng.choice(["Phones", "Chairs", "Paper", "Shoes"], n_demo),
        "Product_Name":     rng.choice(
            ["iPhone", "Chair", "Xerox", "Nike", "Sofa", "Pen"], n_demo),
        "Quantity":         rng.integers(1, 10, n_demo),
        "Unit_Price":       rng.uniform(10, 500, n_demo).round(2),
        "Discount":         rng.uniform(0, 0.5, n_demo).round(2),
        "Shipping_Cost":    rng.uniform(1, 50, n_demo).round(2),
        "Region":           rng.choice(["North", "South", "East", "West"], n_demo),
        "City":             rng.choice(["Lahore", "Karachi", "Islamabad", "Multan"], n_demo),
        "State":            rng.choice(["Punjab", "Sindh", "Federal", "Punjab"], n_demo),
        "Segment":          rng.choice(["Consumer", "Corporate", "Home Office"], n_demo),
    })
    demo_df["Total_Sale"] = (
        demo_df["Quantity"] * demo_df["Unit_Price"] * (1 - demo_df["Discount"])
    ).round(2)
    demo_df["Profit"] = (
        demo_df["Total_Sale"] * (0.20 - demo_df["Discount"])
    ).round(2)
    demo_df.to_csv(DEMO_FILE, index=False)


# ============================================================
# SESSION STATE
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "home"


# ============================================================
# PRE-LOAD DATA + PRE-TRAIN MODELS (runs ONCE, cached forever)
# ============================================================
@st.cache_data(show_spinner=False)
def _load_data(file_path: str):
    df = load_and_prepare(file_path)
    df = add_features(df)
    return df


@st.cache_resource(show_spinner=False)
def _train_models(df_hash: str, _df: pd.DataFrame):
    """Trains all 4 models once and saves them to /models."""
    results, pipes, test_set = md.train_all(_df)
    return results, pipes, test_set


# ---- Pre-load with a nice spinner ----
with st.spinner("⚡ Loading data & training ML models (one-time)…"):
    df = _load_data(str(DEMO_FILE))
    # hash the data to invalidate the cache when data changes
    df_hash = str(hash(str(df.shape) + str(df["Profit"].sum())))
    results, pipes, (X_test, y_test) = _train_models(df_hash, df)

st.success(f"✅ Ready — {len(df):,} rows · 4 models trained")


# ============================================================
# TOP NAVBAR
# ============================================================
PAGES = {
    "home":      "🏠 Home",
    "explorer":  "🔍 Explorer",
    "dashboard": "📈 Dashboard",
    "product":   "🛍 Product",
    "customer":  "👥 Customer",
    "models":    "🤖 Models",
    "predict":   "💰 Predict",
}

nav_cols = st.columns(len(PAGES))
for i, (key, label) in enumerate(PAGES.items()):
    with nav_cols[i]:
        is_active = (st.session_state.page == key)
        if st.button(
            label,
            key=f"navbtn_{key}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.page = key
            st.rerun()

page = st.session_state.page


# ============================================================
# PAGE 1: HOME
# ============================================================
if page == "home":
    st.markdown("""
    <div class="hero">
        <h1>📊 Sales Analytics & Profit Prediction</h1>
        <p>An interactive business intelligence dashboard powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="card">
            <h3>🎯 Business Problem</h3>
            <p>A retail company wants to understand sales performance, identify
            profitable products, categories, regions and salespeople, quantify
            the impact of discounts and delivery time, and predict the profit
            of future orders.</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card">
            <h3>🚀 Objectives</h3>
            <ul>
            <li>Build interactive sales dashboard</li>
            <li>Explore product / region / customer drivers</li>
            <li>Predict order profit using ML</li>
            <li>Recommend actionable strategies</li>
            </ul></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="card">
            <h3>📁 Dataset</h3>
            <p>Rows: <b>{len(df):,}</b><br>
            Columns: <b>{df.shape[1]}</b><br>
            Orders: <b>{df['Order_ID'].nunique():,}</b><br>
            Customers: <b>{df['Customer_ID'].nunique():,}</b></p></div>""",
            unsafe_allow_html=True)

    st.markdown('<div class="section-title">👨‍🎓 Project Info</div>', unsafe_allow_html=True)
    st.info("**Student:** (SHAKEEL AHMED CODE 778197)  •  **Course:** Machine Learning & Deep Learning  •  **Project:** Sales Analytics & Profit Prediction")

    st.markdown('<div class="section-title">🧭 Project Flow</div>', unsafe_allow_html=True)
    steps = [
        ("1️⃣", "Data Understanding", "Shape, dtypes, missing, duplicates"),
        ("2️⃣", "Data Cleaning", "Dates, negatives, discount normalisation"),
        ("3️⃣", "Feature Engineering", "10+ engineered features"),
        ("4️⃣", "EDA", "KPIs, charts, business answers"),
        ("5️⃣", "Regression Modeling", "Linear, DT, RF, Gradient Boosting"),
        ("6️⃣", "Prediction Form", "Real-time profit inference"),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f"""<div class="card">
                <div style="font-size:2rem;">{icon}</div>
                <h3>{title}</h3>
                <p>{desc}</p></div>""", unsafe_allow_html=True)
            st.markdown("")


# ============================================================
# PAGE 2: EXPLORER
# ============================================================
elif page == "explorer":
    st.markdown('<div class="section-title">🔍 Dataset Explorer</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        regions = st.multiselect("Region", sorted(df["Region"].dropna().unique()))
    with c2:
        cats = st.multiselect("Category", sorted(df["Product_Category"].dropna().unique()))
    with c3:
        prods = st.multiselect("Product", sorted(df["Product_Name"].dropna().unique())[:200])
    with c4:
        n_rows = st.slider("Rows preview", 5, 500, 25)

    dr = None
    if "Order_Date" in df.columns and df["Order_Date"].notna().any():
        dr = st.date_input("Order Date range",
                           value=(df["Order_Date"].min().date(),
                                  df["Order_Date"].max().date()))

    view = df.copy()
    if regions: view = view[view["Region"].isin(regions)]
    if cats:    view = view[view["Product_Category"].isin(cats)]
    if prods:   view = view[view["Product_Name"].isin(prods)]
    if dr is not None and len(dr) == 2:
        view = view[(view["Order_Date"].dt.date >= dr[0]) &
                    (view["Order_Date"].dt.date <= dr[1])]

    st.markdown(f"**Filtered rows:** `{len(view):,}`")
    st.dataframe(view.head(n_rows), use_container_width=True)

    st.download_button("⬇️ Download filtered CSV",
                       view.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_data.csv",
                       mime="text/csv")

    with st.expander("📊 Data Understanding"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Shape:**", df.shape)
            st.write("**Missing Values:**")
            st.dataframe(df.isna().sum().rename("missing"))
            st.write("**Duplicates:**", df.duplicated().sum())
        with c2:
            st.write("**Dtypes:**")
            st.dataframe(df.dtypes.astype(str).rename("dtype"))
            st.write("### Answers")
            st.write(f"- Orders: **{df['Order_ID'].nunique()}**")
            st.write(f"- Customers: **{df['Customer_ID'].nunique()}**")
            st.write(f"- Products: **{df['Product_Name'].nunique()}**")
            st.write(f"- Brands: **{df['Brand'].nunique()}**")
        st.write("**Numeric Summary:**")
        st.dataframe(df.describe().T)


# ============================================================
# PAGE 3: DASHBOARD
# ============================================================
elif page == "dashboard":
    st.markdown('<div class="section-title">📈 Sales Dashboard</div>', unsafe_allow_html=True)

    k = an.kpis(df)
    cols = st.columns(4)
    for i, (label, val) in enumerate(k.items()):
        with cols[i % 4]:
            if "Rate" in label:
                disp = f"{val:.2f}%"
            elif isinstance(val, (int, np.integer)):
                disp = f"{val:,}"
            else:
                disp = f"{val:,.2f}"
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{disp}</div>
            </div>""", unsafe_allow_html=True)

    mp = an.monthly_sales_profit(df)
    if len(mp) > 0:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(mp, x="Period", y="Sales", markers=True,
                          title="📅 Monthly Sales",
                          color_discrete_sequence=["#6C5CE7"])
            fig.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.line(mp, x="Period", y="Profit", markers=True,
                          title="💹 Monthly Profit",
                          color_discrete_sequence=["#00CEC9"])
            fig.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid dates found — monthly trend unavailable.")

    cat = an.category_summary(df)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(cat, x="Product_Category", y="Sales", color="Product_Category",
                     title="🏷 Sales by Category")
        fig.update_layout(template="plotly_white", showlegend=False, height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(cat, x="Product_Category", y="Profit", color="Product_Category",
                     title="💰 Profit by Category")
        fig.update_layout(template="plotly_white", showlegend=False, height=360)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 4: PRODUCT
# ============================================================
elif page == "product":
    st.markdown('<div class="section-title">🛍 Product & Regional Analysis</div>', unsafe_allow_html=True)

    tp_sales = an.top_products(df, "Total_Sale")
    tp_profit = an.top_products(df, "Profit")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(tp_sales, x="Total_Sale", y="Product_Name", orientation="h",
                     title="🏆 Top 10 Products by Sales",
                     color="Total_Sale", color_continuous_scale="Purples")
        fig.update_layout(template="plotly_white", height=420, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(tp_profit, x="Profit", y="Product_Name", orientation="h",
                     title="🏆 Top 10 Products by Profit",
                     color="Profit", color_continuous_scale="Teal")
        fig.update_layout(template="plotly_white", height=420, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    prod_stats = (df.groupby("Product_Name")
                  .agg(Sales=("Total_Sale", "sum"), Profit=("Profit", "sum")).reset_index())
    prod_stats["Margin"] = prod_stats["Profit"] / prod_stats["Sales"].replace(0, np.nan)
    high_risk = (prod_stats.sort_values("Sales", ascending=False)
                 .head(30).sort_values("Margin").head(10))
    st.markdown("#### ⚠️ High Sales but Low Profit Products")
    st.dataframe(high_risk, use_container_width=True)

    region = an.region_summary(df)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(region, x="Region", y="Sales", color="Region",
                     title="🌍 Sales by Region")
        fig.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(region, x="Region", y="Profit", color="Region",
                     title="💵 Profit by Region")
        fig.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    brand = (df.groupby("Brand")["Total_Sale"].sum().sort_values(ascending=False)
             .head(10).reset_index())
    fig = px.bar(brand, x="Brand", y="Total_Sale", title="🏷 Top Brands by Revenue",
                 color="Total_Sale", color_continuous_scale="Sunset")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    sp = an.salesperson_summary(df).head(15)
    fig = px.bar(sp, x="Salesperson", y="Profit", color="Profit",
                 title="👔 Salesperson Performance (Top 15)")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 5: CUSTOMER
# ============================================================
elif page == "customer":
    st.markdown('<div class="section-title">👥 Customer & Delivery Analysis</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        g = df.groupby("Customer_Gender")["Total_Sale"].sum().reset_index()
        fig = px.pie(g, names="Customer_Gender", values="Total_Sale",
                     title="🚻 Sales by Gender", hole=0.45,
                     color_discrete_sequence=["#6C5CE7", "#00CEC9"])
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        a = df.groupby("Age_Group", observed=True)["Total_Sale"].sum().reset_index()
        fig = px.bar(a, x="Age_Group", y="Total_Sale", title="🎂 Sales by Age Group",
                     color="Total_Sale", color_continuous_scale="Purples")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Customer_Rating", nbins=20,
                           title="⭐ Customer Rating Distribution",
                           color_discrete_sequence=["#6C5CE7"])
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df, x="Delivery_Days", nbins=20,
                           title="🚚 Delivery Days Distribution",
                           color_discrete_sequence=["#00CEC9"])
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(df, x="Delivery_Status", y="Customer_Rating",
                     title="⭐ Delivery Time vs Rating", color="Delivery_Status")
        fig.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        rr = df.groupby("Product_Category")["Returned"].mean().mul(100).reset_index()
        fig = px.bar(rr, x="Product_Category", y="Returned",
                     title="🔁 Return Rate by Category (%)",
                     color="Returned", color_continuous_scale="Reds")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    fig = px.imshow(df[num_cols].corr(), text_auto=".2f",
                    title="🔥 Correlation Heatmap", aspect="auto",
                    color_continuous_scale="RdBu_r")
    fig.update_layout(template="plotly_white", height=650)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 6: MODELS
# ============================================================
elif page == "models":
    st.markdown('<div class="section-title">🤖 Model Evaluation</div>', unsafe_allow_html=True)

    best = results.iloc[0]
    st.markdown(f"""
    <div class="hero" style="padding:1.6rem; margin-bottom:1.4rem;">
        <h1 style="font-size:1.6rem;">🏆 Best Model: {best['Model']}</h1>
        <p style="margin-top:0.5rem;">
            R² = <b>{best['R2']:.4f}</b>  •  
            MAE = <b>${best['MAE']:,.2f}</b>  •  
            RMSE = <b>${best['RMSE']:,.2f}</b>  •  
            MAPE = <b>{best['MAPE (%)']:.2f}%</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 All Metrics — Model Comparison</div>',
                unsafe_allow_html=True)

    display_cols = ["Model", "MAE", "MSE", "RMSE", "R2",
                    "Explained_Var", "Max_Error", "MAPE (%)",
                    "Train R2", "Test R2", "Overfit Gap"]

    st.dataframe(
        results[display_cols].style
        .format({
            "MAE": "{:.2f}",
            "MSE": "{:.2f}",
            "RMSE": "{:.2f}",
            "R2": "{:.4f}",
            "Explained_Var": "{:.4f}",
            "Max_Error": "{:.2f}",
            "MAPE (%)": "{:.2f}",
            "Train R2": "{:.4f}",
            "Test R2": "{:.4f}",
            "Overfit Gap": "{:.4f}",
        })
        .highlight_max(subset=["R2", "Explained_Var", "Train R2"], color="#D4F5E9")
        .highlight_min(subset=["MAE", "MSE", "RMSE", "MAPE (%)", "Overfit Gap"],
                       color="#D4F5E9"),
        use_container_width=True,
        height=280,
    )

    with st.expander("ℹ️ What does each metric mean?"):
        st.markdown("""
        | Metric | Meaning | Better when… |
        |---|---|---|
        | **MAE** | Average absolute error (in $) | Lower ↓ |
        | **MSE** | Average squared error | Lower ↓ |
        | **RMSE** | Square-root of MSE | Lower ↓ |
        | **R²** | % variance explained | Higher ↑ |
        | **Explained Variance** | Variance captured | Higher ↑ |
        | **Max Error** | Worst-case prediction | Lower ↓ |
        | **MAPE (%)** | Average % error | Lower ↓ |
        | **Train/Test R² gap** | Overfitting indicator | Close to 0 |
        """)

    st.markdown('<div class="section-title">📈 Metrics Visualised</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(results, x="Model", y="R2", color="Model",
                     title="R² Score (higher is better)",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(results, x="Model", y="MAPE (%)", color="Model",
                     title="MAPE % (lower is better)",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    metric_melt = results.melt(
        id_vars="Model",
        value_vars=["MAE", "RMSE", "Max_Error"],
        var_name="Metric", value_name="Value",
    )
    fig = px.bar(metric_melt, x="Model", y="Value", color="Metric",
                 barmode="group", title="Error Metrics (lower is better)")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # ---- FIXED: value_name changed from "R2" to "R2_Score" ----
    r2_melt = results.melt(
        id_vars="Model",
        value_vars=["Train R2", "Test R2"],
        var_name="Set",
        value_name="R2_Score",          # ✅ naya naam (R2 already exists)
    )
    fig = px.bar(r2_melt, x="Model", y="R2_Score", color="Set",   # ✅ y bhi update
                 barmode="group",
                 title="Train vs Test R²  (large gap = overfitting)")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">🔎 Per-Model Detailed Breakdown</div>',
                unsafe_allow_html=True)

    selected = st.selectbox("Choose a model to inspect", results["Model"].tolist())
    row = results[results["Model"] == selected].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">R² Score</div>
            <div class="kpi-value">{row['R2']:.4f}</div></div>""",
            unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="kpi-card" style="border-left-color:#00CEC9;">
            <div class="kpi-label">MAE</div>
            <div class="kpi-value">${row['MAE']:,.2f}</div></div>""",
            unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="kpi-card" style="border-left-color:#FDCB6E;">
            <div class="kpi-label">RMSE</div>
            <div class="kpi-value">${row['RMSE']:,.2f}</div></div>""",
            unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="kpi-card" style="border-left-color:#FF6B6B;">
            <div class="kpi-label">MAPE</div>
            <div class="kpi-value">{row['MAPE (%)']:.2f}%</div></div>""",
            unsafe_allow_html=True)

    pipe = pipes[selected]
    y_pred = pipe.predict(X_test)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(x=y_test, y=y_pred, opacity=0.5,
                         labels={"x": "Actual Profit", "y": "Predicted Profit"},
                         title=f"🎯 Actual vs Predicted — {selected}",
                         color_discrete_sequence=["#6C5CE7"])
        lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        fig.add_shape(type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1],
                      line=dict(color="#FDCB6E", dash="dash", width=2))
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        res = y_test - y_pred
        fig = px.histogram(res, nbins=40,
                           title=f"📉 Residuals — {selected}",
                           color_discrete_sequence=["#00CEC9"])
        fig.add_vline(x=0, line_dash="dash", line_color="#FF6B6B")
        fig.update_layout(template="plotly_white",
                          xaxis_title="Residual (Actual − Predicted)",
                          yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    try:
        model = pipe.named_steps["model"]
        prep = pipe.named_steps["prep"]
        cat_names = prep.transformers_[0][1].get_feature_names_out(
            prep.transformers_[0][2])
        num_names = prep.transformers_[1][2]
        names = list(cat_names) + list(num_names)

        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            fi = pd.DataFrame({"Feature": names, "Importance": imp}) \
                   .sort_values("Importance", ascending=False).head(20)
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                         title=f"🔍 Top 20 Feature Importances — {selected}",
                         color="Importance", color_continuous_scale="Purples")
            fig.update_layout(template="plotly_white", height=600,
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"📌 {selected} does not provide feature importance.")
    except Exception as e:
        st.info(f"Feature importance unavailable: {e}")


# ============================================================
# PAGE 7: PREDICT
# ============================================================
elif page == "predict":
    st.markdown('<div class="section-title">💰 Predict Order Profit</div>',
                unsafe_allow_html=True)
    st.markdown("Enter order details to estimate **expected profit**.")

    best_model_name = st.selectbox("Select model", list(pipes.keys()))

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            cat = st.selectbox("Product Category",
                               sorted(df["Product_Category"].dropna().unique()))
            subcat = st.selectbox("Sub-Category",
                                  sorted(df["Sub_Category"].dropna().unique()))
            region = st.selectbox("Region",
                                  sorted(df["Region"].dropna().unique()))
            city = st.selectbox("City",
                                sorted(df["City"].dropna().unique()))
        with c2:
            qty = st.number_input("Quantity", 1, 100, 2)
            unit_price = st.number_input("Unit Price", 0.0, 10000.0, 50.0)
            discount = st.slider("Discount", 0.0, 0.8, 0.1, 0.01)
            ship_cost = st.number_input("Shipping Cost", 0.0, 1000.0, 5.0)
        with c3:
            delivery = st.number_input("Delivery Days", 0, 30, 4)
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", 18, 90, 35)
            payment = st.selectbox("Payment",
                                   ["Credit Card", "PayPal",
                                    "Cash on Delivery", "Debit Card"])
            salesperson = st.selectbox("Salesperson",
                                       sorted(df["Salesperson"].unique()))
            ship_mode = st.selectbox("Ship Mode",
                                     sorted(df["Ship_Mode"].dropna().unique()))

        submit = st.form_submit_button("🔮 Predict Profit")

    if submit:
        row = {
            "Customer_Gender": gender, "Customer_Age": age, "City": city,
            "Region": region, "Product_Category": cat, "Product_Name": "Unknown",
            "Brand": "Unknown", "Quantity": qty, "Unit_Price": unit_price,
            "Discount": discount, "Shipping_Cost": ship_cost,
            "Payment_Method": payment, "Delivery_Days": delivery,
            "Salesperson": salesperson, "Sub_Category": subcat,
            "Segment": "Consumer", "Ship_Mode": ship_mode, "State": "Unknown",
            "Delivery_Status": ("Fast" if delivery <= 2 else
                                "Normal" if delivery <= 5 else "Late"),
            "Order_Size_Category": ("Small" if qty <= 2 else
                                    "Medium" if qty <= 5 else
                                    "Large" if qty <= 10 else "Bulk"),
            "Age_Group": ("<25" if age < 25 else "25-34" if age < 35 else
                          "35-44" if age < 45 else "45-54" if age < 55 else "55+"),
            "Order_Year": 2024, "Order_Month": 1, "Quarter": 1,
            "Discount_Amount": unit_price * qty * discount,
            "Revenue_Before_Discount": unit_price * qty,
        }
        X_new = pd.DataFrame([row])
        pred = float(pipes[best_model_name].predict(X_new)[0])

        category = ("Loss" if pred < 0 else
                    "Low Profit" if pred < 20 else
                    "Medium Profit" if pred < 100 else "High Profit")
        emoji = {"Loss": "🔴", "Low Profit": "🟡",
                 "Medium Profit": "🟢", "High Profit": "💎"}[category]

        st.markdown(f"""
        <div class="hero" style="text-align:center; padding:2rem;">
            <h1>${pred:,.2f}</h1>
            <p>Predicted Profit — {emoji} {category}</p>
        </div>
        """, unsafe_allow_html=True)

        if pred < 0:
            st.error("⚠️ **Loss Warning:** Predicted profit is negative. "
                     "Consider reducing discount or renegotiating cost.")
        else:
            st.success("✅ This order is expected to be profitable.")