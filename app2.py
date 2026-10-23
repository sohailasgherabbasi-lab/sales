import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from utils.data_loader import load_and_prepare
from utils.feature_engineering import add_features
from utils import analytics as an
from utils import modeling as md

# ================= CONFIG =================
st.set_page_config(
    page_title="Sales Analytics & Profit Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================= STYLING =================
st.markdown("""
<style>
/* Main container */
.main {background: linear-gradient(180deg, #0E1117 0%, #151829 100%);}
h1, h2, h3 {color:#FAFAFA !important; font-family: 'Segoe UI', sans-serif;}
[data-testid="stMetricValue"] {color:#A29BFE !important; font-weight:700;}
[data-testid="stMetricLabel"] {color:#B2BEC3 !important;}
.stButton>button {
    background: linear-gradient(90deg,#6C5CE7,#A29BFE);
    color:white; border-radius:12px; border:none;
    padding:0.6rem 1.4rem; font-weight:600;
    box-shadow:0 4px 14px rgba(108,92,231,.4);
}
.stButton>button:hover {transform:translateY(-2px); transition:.2s;}
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1A1D29,#0E1117);
}
.card {
    background: linear-gradient(135deg,#1E2235,#151829);
    border:1px solid #2A2F45;
    padding:1.2rem; border-radius:16px;
    box-shadow:0 8px 24px rgba(0,0,0,.35);
}
.big-title {
    font-size:2.6rem; font-weight:800;
    background:linear-gradient(90deg,#6C5CE7,#00CEC9,#FDCB6E);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.sub {color:#B2BEC3; font-size:1.05rem;}
</style>
""", unsafe_allow_html=True)


# ================= DATA CACHE =================
@st.cache_data(show_spinner=False)
def get_data(file_bytes):
    df = load_and_prepare(file_bytes)
    df = add_features(df)
    return df


@st.cache_resource(show_spinner=True)
def get_models(df):
    return md.train_all(df)


# ================= SIDEBAR =================
st.sidebar.markdown("## 📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔍 Dataset Explorer", "📈 Sales Dashboard",
     "🛍 Product & Regional", "👥 Customer & Delivery",
     "🤖 Model Evaluation", "💰 Profit Prediction"],
)

uploaded = st.sidebar.file_uploader("Upload Global Superstore CSV", type=["csv"])
st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ in Streamlit")

if uploaded is None:
    st.warning("⬅️ Upload `Global Superstore.csv` from the sidebar to begin.")
    st.stop()

df = get_data(uploaded)

# ================= PAGE 1: HOME =================
if page == "🏠 Home":
    st.markdown('<div class="big-title">Sales Analytics & Profit Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">An interactive business intelligence dashboard powered by Machine Learning</div>', unsafe_allow_html=True)
    st.markdown("")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><h3>🎯 Business Problem</h3>'
                    '<p>A retail company wants to understand sales performance, '
                    'identify profitable products, categories, regions and salespeople, '
                    'quantify the impact of discounts and delivery time, and predict '
                    'the profit of future orders.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h3>🎯 Objectives</h3>'
                    '<ul><li>Build interactive sales dashboard</li>'
                    '<li>Explore product / region / customer drivers</li>'
                    '<li>Predict order profit using ML</li>'
                    '<li>Recommend actionable strategies</li></ul></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><h3>📁 Dataset</h3>'
                    f'<p>Rows: <b>{len(df):,}</b><br>'
                    f'Columns: <b>{df.shape[1]}</b><br>'
                    f'Orders: <b>{df["Order_ID"].nunique():,}</b><br>'
                    f'Customers: <b>{df["Customer_ID"].nunique():,}</b></p></div>',
                    unsafe_allow_html=True)

    st.markdown("### 👨‍🎓 Student & Course")
    st.info("**Student:** (Your Name)  •  **Course:** Machine Learning & Deep Learning  •  **Project:** Sales Analytics & Profit Prediction")

    st.markdown("### 🚀 Project Flow")
    st.markdown("""
    1. **Data Understanding** → shape, dtypes, missing, duplicates
    2. **Data Cleaning** → dates, negatives, discount normalisation
    3. **Feature Engineering** → 10+ engineered features
    4. **EDA** → KPIs, charts, business answers
    5. **Regression Modeling** → Linear, Decision Tree, Random Forest, Gradient Boosting
    6. **Prediction Form** → real-time profit inference
    """)

# ================= PAGE 2: DATASET EXPLORER =================
elif page == "🔍 Dataset Explorer":
    st.title("🔍 Dataset Explorer")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        regions = st.multiselect("Region", sorted(df["Region"].dropna().unique()))
    with c2:
        cats = st.multiselect("Category", sorted(df["Product_Category"].dropna().unique()))
    with c3:
        prods = st.multiselect("Product", sorted(df["Product_Name"].dropna().unique())[:200])
    with c4:
        if "Order_Date" in df.columns:
            dr = st.date_input("Order Date range",
                               value=(df["Order_Date"].min().date(), df["Order_Date"].max().date()))

    n_rows = st.slider("Rows to preview", 5, 500, 25)

    view = df.copy()
    if regions:
        view = view[view["Region"].isin(regions)]
    if cats:
        view = view[view["Product_Category"].isin(cats)]
    if prods:
        view = view[view["Product_Name"].isin(prods)]
    if "Order_Date" in df.columns and len(dr) == 2:
        view = view[(view["Order_Date"].dt.date >= dr[0]) & (view["Order_Date"].dt.date <= dr[1])]

    st.markdown(f"**Filtered rows:** `{len(view):,}`")
    st.dataframe(view.head(n_rows), use_container_width=True)

    st.download_button("⬇️ Download filtered CSV",
                       view.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_superstore.csv",
                       mime="text/csv")

    with st.expander("📊 Data Understanding"):
        st.write("**Shape:**", df.shape)
        st.write("**Columns:**", list(df.columns))
        st.write("**Dtypes:**")
        st.dataframe(df.dtypes.astype(str).rename("dtype"))
        st.write("**Missing Values:**")
        st.dataframe(df.isna().sum().rename("missing"))
        st.write("**Duplicates:**", df.duplicated().sum())
        st.write("**Numeric summary:**")
        st.dataframe(df.describe().T)
        st.write("### Business answers")
        st.write(f"- Orders: **{df['Order_ID'].nunique()}**")
        st.write(f"- Customers: **{df['Customer_ID'].nunique()}**")
        st.write(f"- Products: **{df['Product_Name'].nunique()}**")
        st.write(f"- Brands: **{df['Brand'].nunique()}**")
        if "Order_Date" in df.columns:
            st.write(f"- Date range: **{df['Order_Date'].min().date()} → {df['Order_Date'].max().date()}**")

# ================= PAGE 3: SALES DASHBOARD =================
elif page == "📈 Sales Dashboard":
    st.title("📈 Sales Dashboard")
    k = an.kpis(df)
    cols = st.columns(4)
    for i, (label, val) in enumerate(k.items()):
        with cols[i % 4]:
            if "Rate" in label:
                st.metric(label, f"{val:.2f}%")
            elif isinstance(val, (int, np.integer)):
                st.metric(label, f"{val:,}")
            else:
                st.metric(label, f"{val:,.2f}")

    mp = an.monthly_sales_profit(df)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(mp, x="Period", y="Sales", markers=True,
                      title="📅 Monthly Sales",
                      color_discrete_sequence=["#6C5CE7"])
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(mp, x="Period", y="Profit", markers=True,
                      title="💹 Monthly Profit",
                      color_discrete_sequence=["#00CEC9"])
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

    cat = an.category_summary(df)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(cat, x="Product_Category", y="Sales", color="Product_Category",
                     title="🏷 Sales by Category")
        fig.update_layout(template="plotly_dark", showlegend=False, height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(cat, x="Product_Category", y="Profit", color="Product_Category",
                     title="💰 Profit by Category")
        fig.update_layout(template="plotly_dark", showlegend=False, height=360)
        st.plotly_chart(fig, use_container_width=True)

# ================= PAGE 4: PRODUCT & REGIONAL =================
elif page == "🛍 Product & Regional":
    st.title("🛍 Product & Regional Analysis")

    tp_sales = an.top_products(df, "Total_Sale")
    tp_profit = an.top_products(df, "Profit")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(tp_sales, x="Total_Sale", y="Product_Name", orientation="h",
                     title="🏆 Top 10 Products by Sales",
                     color="Total_Sale", color_continuous_scale="Viridis")
        fig.update_layout(template="plotly_dark", height=420, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(tp_profit, x="Profit", y="Product_Name", orientation="h",
                     title="🏆 Top 10 Products by Profit",
                     color="Profit", color_continuous_scale="Teal")
        fig.update_layout(template="plotly_dark", height=420, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # High sales but low profit
    prod_stats = (df.groupby("Product_Name")
                  .agg(Sales=("Total_Sale", "sum"), Profit=("Profit", "sum")).reset_index())
    prod_stats["Margin"] = prod_stats["Profit"] / prod_stats["Sales"].replace(0, np.nan)
    high_risk = prod_stats.sort_values("Sales", ascending=False).head(30).sort_values("Margin").head(10)
    st.markdown("#### ⚠️ High Sales but Low Profit Products")
    st.dataframe(high_risk, use_container_width=True)

    region = an.region_summary(df)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(region, x="Region", y="Sales", color="Region",
                     title="🌍 Sales by Region")
        fig.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(region, x="Region", y="Profit", color="Region",
                     title="💵 Profit by Region")
        fig.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    brand = (df.groupby("Brand")["Total_Sale"].sum().sort_values(ascending=False)
             .head(10).reset_index())
    fig = px.bar(brand, x="Brand", y="Total_Sale", title="🏷 Top Brands by Revenue",
                 color="Total_Sale", color_continuous_scale="Sunset")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    sp = an.salesperson_summary(df).head(15)
    fig = px.bar(sp, x="Salesperson", y="Profit", color="Profit",
                 title="👔 Salesperson Performance (Top 15)")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ================= PAGE 5: CUSTOMER & DELIVERY =================
elif page == "👥 Customer & Delivery":
    st.title("👥 Customer & Delivery Analysis")

    c1, c2 = st.columns(2)
    with c1:
        g = df.groupby("Customer_Gender")["Total_Sale"].sum().reset_index()
        fig = px.pie(g, names="Customer_Gender", values="Total_Sale",
                     title="🚻 Sales by Gender", hole=0.45)
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        a = df.groupby("Age_Group", observed=True)["Total_Sale"].sum().reset_index()
        fig = px.bar(a, x="Age_Group", y="Total_Sale", title="🎂 Sales by Age Group",
                     color="Total_Sale", color_continuous_scale="Purples")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Customer_Rating", nbins=20,
                           title="⭐ Customer Rating Distribution")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df, x="Delivery_Days", nbins=20,
                           title="🚚 Delivery Days Distribution")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(df, x="Delivery_Status", y="Customer_Rating",
                     title="⭐ Delivery Time vs Rating", color="Delivery_Status")
        fig.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        rr = df.groupby("Product_Category")["Returned"].mean().mul(100).reset_index()
        fig = px.bar(rr, x="Product_Category", y="Returned",
                     title="🔁 Return Rate by Category (%)",
                     color="Returned", color_continuous_scale="Reds")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    top_cust = (df.groupby("Customer_Name")["Total_Sale"].sum()
                .sort_values(ascending=False).head(15).reset_index())
    st.markdown("#### 💎 Top 15 Customers")
    st.dataframe(top_cust, use_container_width=True)

    # Correlation heatmap
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    fig = px.imshow(df[num_cols].corr(), text_auto=".2f",
                    title="🔥 Correlation Heatmap", aspect="auto",
                    color_continuous_scale="RdBu_r")
    fig.update_layout(template="plotly_dark", height=650)
    st.plotly_chart(fig, use_container_width=True)

# ================= PAGE 6: MODEL EVALUATION =================
elif page == "🤖 Model Evaluation":
    st.title("🤖 Model Evaluation")
    with st.spinner("Training models (Linear, DT, RF, GB)…"):
        results, pipes, (X_test, y_test) = get_models(df)

    st.subheader("📊 Model Comparison Table")
    st.dataframe(results.style.format({
        "MAE": "{:.2f}", "MSE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}"
    }).highlight_max(subset=["R2"], color="#00CEC9")
        .highlight_min(subset=["MAE", "RMSE"], color="#00CEC9"),
        use_container_width=True)

    fig = px.bar(results.melt(id_vars="Model",
                              value_vars=["MAE", "RMSE", "R2"]),
                 x="Model", y="value", color="variable", barmode="group",
                 title="📈 Model Comparison")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    best = results.iloc[0]["Model"]
    st.success(f"🏆 Best model: **{best}** (R² = {results.iloc[0]['R2']:.4f})")

    pipe = pipes[best]
    y_pred = pipe.predict(X_test)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(x=y_test, y=y_pred, opacity=0.5,
                         labels={"x": "Actual", "y": "Predicted"},
                         title="🎯 Actual vs Predicted")
        lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        fig.add_shape(type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1],
                      line=dict(color="#FDCB6E", dash="dash"))
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        res = y_test - y_pred
        fig = px.histogram(res, nbins=40, title="📉 Residual Distribution")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    try:
        model = pipe.named_steps["model"]
        prep = pipe.named_steps["prep"]
        cat_names = prep.transformers_[0][1].get_feature_names_out(
            prep.transformers_[0][2])
        num_names = prep.transformers_[1][2]
        names = list(cat_names) + list(num_names)
        imp = model.feature_importances_
        fi = pd.DataFrame({"Feature": names, "Importance": imp}) \
               .sort_values("Importance", ascending=False).head(20)
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                     title="🔍 Top 20 Feature Importances",
                     color="Importance", color_continuous_scale="Viridis")
        fig.update_layout(template="plotly_dark", height=600,
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info("Feature importance unavailable for this model.")

# ================= PAGE 7: PROFIT PREDICTION =================
elif page == "💰 Profit Prediction":
    st.title("💰 Predict Order Profit")
    st.markdown("Enter order details to estimate **expected profit**.")

    with st.spinner("Loading model…"):
        try:
            pipes = {n: md.load_model(n) for n in
                     ["Linear_Regression", "Decision_Tree", "Random_Forest",
                      "Gradient_Boosting"]}
        except Exception:
            results, pipes, _ = get_models(df)

    best_model_name = st.selectbox("Select model", list(pipes.keys()))

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            cat = st.selectbox("Product Category", sorted(df["Product_Category"].dropna().unique()))
            subcat = st.selectbox("Sub-Category", sorted(df["Sub_Category"].dropna().unique()))
            region = st.selectbox("Region", sorted(df["Region"].dropna().unique()))
            city = st.selectbox("City", sorted(df["City"].dropna().unique()))
        with c2:
            qty = st.number_input("Quantity", 1, 100, 2)
            unit_price = st.number_input("Unit Price", 0.0, 10000.0, 50.0)
            discount = st.slider("Discount", 0.0, 0.8, 0.1, 0.01)
            ship_cost = st.number_input("Shipping Cost", 0.0, 1000.0, 5.0)
        with c3:
            delivery = st.number_input("Delivery Days", 0, 30, 4)
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", 18, 90, 35)
            payment = st.selectbox("Payment", ["Credit Card", "PayPal",
                                                "Cash on Delivery", "Debit Card"])
            salesperson = st.selectbox("Salesperson", sorted(df["Salesperson"].unique()))
            ship_mode = st.selectbox("Ship Mode", sorted(df["Ship_Mode"].dropna().unique()))

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
        <div class="card" style="text-align:center;">
          <h2>Predicted Profit</h2>
          <h1 style="font-size:3.5rem;color:#A29BFE;">${pred:,.2f}</h1>
          <h3>{emoji} {category}</h3>
        </div>
        """, unsafe_allow_html=True)

        if pred < 0:
            st.error("⚠️ **Loss Warning:** Predicted profit is negative. "
                     "Consider reducing discount or renegotiating cost.")
        else:
            st.success("✅ This order is expected to be profitable.")