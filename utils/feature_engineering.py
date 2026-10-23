import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    if "Order_Date" in d.columns:
        d["Order_Year"] = d["Order_Date"].dt.year
        d["Order_Month"] = d["Order_Date"].dt.month
        d["Month_Name"] = d["Order_Date"].dt.strftime("%b")
        d["Quarter"] = d["Order_Date"].dt.quarter
        d["Day_Name"] = d["Order_Date"].dt.day_name()

    if "Customer_Age" in d.columns:
        d["Age_Group"] = pd.cut(
            d["Customer_Age"],
            bins=[0, 25, 35, 45, 55, 100],
            labels=["<25", "25-34", "35-44", "45-54", "55+"],
        )

    d["Profit_Margin"] = np.where(
        d["Total_Sale"] != 0, (d["Profit"] / d["Total_Sale"]) * 100, 0
    )
    d["Discount_Amount"] = d["Unit_Price"] * d["Quantity"] * d["Discount"]
    d["Revenue_Before_Discount"] = d["Total_Sale"] + d["Discount_Amount"]

    if "Delivery_Days" in d.columns:
        d["Delivery_Status"] = pd.cut(
            d["Delivery_Days"],
            bins=[-1, 2, 5, 100],
            labels=["Fast", "Normal", "Late"],
        )

    d["Order_Size_Category"] = pd.cut(
        d["Quantity"],
        bins=[0, 2, 5, 10, 1000],
        labels=["Small", "Medium", "Large", "Bulk"],
    )
    return d