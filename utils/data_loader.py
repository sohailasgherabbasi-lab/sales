import pandas as pd
import numpy as np
import io


# ============================================================
# Robust date parser
# ============================================================
def _parse_any_date(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    first = s.str.split().str[0]
    num = pd.to_numeric(first, errors="coerce")
    if num.notna().mean() > 0.5 and (num.fillna(0) > 1000).mean() > 0.5:
        return pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(s, errors="coerce", format="mixed")


# ============================================================
# Smart case-insensitive column matcher
# ============================================================
def _find_column(df, candidates):
    """Return the first column whose lowercase name matches any candidate."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# ============================================================
# Flexible loader — works for ANY sales dataset
# ============================================================
def load_and_prepare(file):
    """
    Accepts a file object OR a path string.
    Works with Global Superstore OR any general sales CSV.
    Auto-maps common columns; synthesises missing ones safely.
    """
    if isinstance(file, (str, bytes)):
        df = pd.read_csv(file)
    else:
        # streamlit UploadedFile — rewind just in case
        try:
            file.seek(0)
        except Exception:
            pass
        raw = file.read()
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")

    df.columns = [c.strip() for c in df.columns]

    # ---------------- Column mapping (case-insensitive) ----------------
    mapping_rules = {
        "Order_ID":      ["order id", "order_id", "orderid", "id", "invoice", "invoice no"],
        "Order_Date":    ["order date", "order_date", "date", "orderdate", "purchase date"],
        "Customer_ID":   ["customer id", "customer_id", "customerid", "cust id", "client id"],
        "Customer_Name": ["customer name", "customer_name", "customer", "name", "client"],
        "Product_Category": ["category", "product_category", "product category", "type"],
        "Sub_Category":  ["sub-category", "sub_category", "subcategory", "sub category"],
        "Product_Name":  ["product name", "product_name", "product", "item", "item name"],
        "Profit":        ["profit", "profit ($)", "gross profit", "net profit"],
        "Quantity":      ["quantity", "qty", "units", "order quantity"],
        "Discount":      ["discount", "discount (%)", "discount rate", "disc"],
        "Shipping_Cost": ["shipping cost", "shipping_cost", "freight", "delivery cost", "ship cost"],
        "Total_Sale":    ["sales", "total_sale", "revenue", "amount", "total amount",
                          "sale", "total sales", "total_sale_amount"],
        "Region":        ["region", "zone", "area", "territory"],
        "City":          ["city", "town"],
        "State":         ["state", "province", "governorate"],
        "Segment":       ["segment", "customer segment", "customer type"],
        "Ship_Date":     ["ship date", "ship_date", "shipped date", "delivery date"],
        "Ship_Mode":     ["ship mode", "ship_mode", "shipping mode", "shipping method"],
    }

    for target, candidates in mapping_rules.items():
        if target in df.columns:
            continue
        found = _find_column(df, candidates)
        if found and found != target:
            df = df.rename(columns={found: target})

    # ---------------- Date parsing ----------------
    for col in ["Order_Date", "Ship_Date"]:
        if col in df.columns:
            parsed = _parse_any_date(df[col])
            if parsed.notna().mean() < 0.1:
                df = df.drop(columns=[col])
            else:
                df[col] = parsed

    # ---------------- Ensure numeric core columns ----------------
    for col in ["Profit", "Total_Sale", "Quantity", "Discount",
                "Shipping_Cost", "Unit_Price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---------------- If no Profit → estimate from Sales & Discount ----------------
    if "Profit" not in df.columns and "Total_Sale" in df.columns:
        disc = df["Discount"].fillna(0) if "Discount" in df.columns else 0
        df["Profit"] = df["Total_Sale"] * (0.20 - disc)

    # ---------------- If no Total_Sale but have Price*Qty ----------------
    if "Total_Sale" not in df.columns:
        if "Unit_Price" in df.columns and "Quantity" in df.columns:
            df["Total_Sale"] = df["Unit_Price"] * df["Quantity"]
        elif "Profit" in df.columns:
            df["Total_Sale"] = np.where(df["Profit"] > 0, df["Profit"] * 5, np.abs(df["Profit"]))

    # ---------------- Unit Price ----------------
    if "Unit_Price" not in df.columns:
        if "Quantity" in df.columns and "Total_Sale" in df.columns:
            df["Unit_Price"] = np.where(df["Quantity"] > 0,
                                        df["Total_Sale"] / df["Quantity"], 0)
        else:
            df["Unit_Price"] = 0.0
    else:
        df["Unit_Price"] = df["Unit_Price"].fillna(0)

    # ---------------- Quantity ----------------
    if "Quantity" not in df.columns:
        df["Quantity"] = 1

    # ---------------- Discount normalisation ----------------
    if "Discount" not in df.columns:
        df["Discount"] = 0.0
    elif df["Discount"].max() > 1:
        df["Discount"] = df["Discount"] / 100.0

    # ---------------- Shipping cost ----------------
    if "Shipping_Cost" not in df.columns:
        df["Shipping_Cost"] = 0.0

    # ---------------- Brand ----------------
    if "Brand" not in df.columns:
        if "Product_Name" in df.columns:
            df["Brand"] = df["Product_Name"].astype(str).str.split().str[0]
        else:
            df["Brand"] = "Unknown"

    # ---------------- Product category ----------------
    if "Product_Category" not in df.columns:
        if "Sub_Category" in df.columns:
            df["Product_Category"] = df["Sub_Category"]
        else:
            df["Product_Category"] = "General"

    # ---------------- Product name ----------------
    if "Product_Name" not in df.columns:
        df["Product_Name"] = "Product"

    # ---------------- Region / City / State ----------------
    if "Region" not in df.columns:
        df["Region"] = "Unknown"
    if "City" not in df.columns:
        df["City"] = "Unknown"
    if "State" not in df.columns:
        df["State"] = "Unknown"
    if "Segment" not in df.columns:
        df["Segment"] = "Consumer"
    if "Ship_Mode" not in df.columns:
        df["Ship_Mode"] = "Standard"

    # ---------------- Delivery Days ----------------
    if {"Order_Date", "Ship_Date"}.issubset(df.columns):
        df["Delivery_Days"] = (df["Ship_Date"] - df["Order_Date"]).dt.days.clip(lower=0)
    else:
        df["Delivery_Days"] = np.nan

    # ---------------- Synthesize missing fields ----------------
    rng = np.random.default_rng(42)
    n = len(df)

    if "Customer_Gender" not in df.columns:
        df["Customer_Gender"] = rng.choice(["Male", "Female"], n)
    if "Customer_Age" not in df.columns:
        df["Customer_Age"] = rng.integers(18, 70, n)
    if "Payment_Method" not in df.columns:
        df["Payment_Method"] = rng.choice(
            ["Credit Card", "PayPal", "Cash on Delivery", "Debit Card"], n)
    if "Salesperson" not in df.columns:
        df["Salesperson"] = rng.choice([f"Rep_{i:02d}" for i in range(1, 21)], n)
    if "Customer_Rating" not in df.columns:
        df["Customer_Rating"] = np.round(rng.uniform(1, 5, n), 1)

    # ---------------- Returned ----------------
    late = (df["Delivery_Days"].fillna(0) > 5).astype(int)
    bad = (df["Profit"] < 0).astype(int)
    prob = np.clip(0.05 + 0.15 * late + 0.10 * bad, 0, 1)
    df["Returned"] = (rng.random(n) < prob).astype(int)

    # ---------------- Order_ID / Customer_ID defaults ----------------
    if "Order_ID" not in df.columns:
        df["Order_ID"] = [f"ORD-{i:06d}" for i in range(1, n + 1)]
    if "Customer_ID" not in df.columns:
        df["Customer_ID"] = [f"CUST-{i:05d}" for i in range(1, n + 1)]
    if "Customer_Name" not in df.columns:
        df["Customer_Name"] = df["Customer_ID"]

    # ---------------- Clean ----------------
    subset = [c for c in ["Order_ID", "Product_ID"] if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="first")

    # Remove rows where target is not available
    df = df.dropna(subset=["Profit"]).reset_index(drop=True)
    if "Total_Sale" in df.columns:
        df = df.dropna(subset=["Total_Sale"]).reset_index(drop=True)

    return df