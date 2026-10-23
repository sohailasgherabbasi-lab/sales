import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error,
)


# ============================================================
# Column groups
# ============================================================
DROP_FROM_FEATURES = [
    "Order_ID", "Customer_ID", "Customer_Name", "Customer_Rating",
    "Returned", "Total_Sale", "Profit", "Product_ID", "Row ID",
    "ji_lu-shu", "Ship_Date", "Order_Date", "Profit_Margin",
]

CAT_COLS = [
    "Customer_Gender", "City", "Region", "Product_Category",
    "Product_Name", "Brand", "Payment_Method", "Salesperson",
    "Sub_Category", "Segment", "Ship_Mode", "State",
    "Delivery_Status", "Order_Size_Category", "Age_Group",
]

NUM_COLS = [
    "Customer_Age", "Quantity", "Unit_Price", "Discount",
    "Shipping_Cost", "Delivery_Days", "Order_Year",
    "Order_Month", "Quarter", "Discount_Amount",
    "Revenue_Before_Discount",
]


# ============================================================
# Build feature matrix
# ============================================================
def build_matrix(df):
    feats = [c for c in CAT_COLS + NUM_COLS if c in df.columns]
    X = df[feats].copy()
    y = df["Profit"]

    for c in CAT_COLS:
        if c in X:
            X[c] = X[c].astype(str)
    for c in NUM_COLS:
        if c in X:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X, y, feats


def preprocessor(cats, nums):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", max_categories=30), cats),
        ("num", StandardScaler(), nums),
    ])


# ============================================================
# Compute all metrics for one model
# ============================================================
def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    maxerr = max_error(y_true, y_pred)

    # MAPE — safe version (avoid division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = np.nan

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Explained_Var": float(evs),
        "Max_Error": float(maxerr),
        "MAPE (%)": float(mape) if not np.isnan(mape) else 0.0,
    }


# ============================================================
# Train all models, return results + pipelines + test data
# ============================================================
def train_all(df: pd.DataFrame, save_dir="models"):
    Path(save_dir).mkdir(exist_ok=True)

    X, y, _ = build_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cats = [c for c in CAT_COLS if c in X.columns]
    nums = [c for c in NUM_COLS if c in X.columns]
    prep = preprocessor(cats, nums)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree":     DecisionTreeRegressor(max_depth=12, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=120, max_depth=15,
                                                   n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=120,
                                                       random_state=42),
    }

    results, pipelines = [], {}

    for name, m in models.items():
        pipe = Pipeline([("prep", prep), ("model", m)])
        pipe.fit(X_train, y_train)

        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)

        test_metrics = _metrics(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        results.append({
            "Model":           name,
            **test_metrics,
            "Train R2":        float(train_r2),
            "Test R2":         float(test_metrics["R2"]),
            "Overfit Gap":     float(train_r2 - test_metrics["R2"]),
        })

        pipelines[name] = pipe
        joblib.dump(pipe, Path(save_dir) / f"{name.replace(' ', '_')}.pkl")

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    return results_df, pipelines, (X_test, y_test)


def load_model(name, save_dir="models"):
    return joblib.load(Path(save_dir) / f"{name.replace(' ', '_')}.pkl")