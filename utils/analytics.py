import pandas as pd
import numpy as np


def kpis(df: pd.DataFrame) -> dict:
    return {
        "Total Sales": df["Total_Sale"].sum(),
        "Total Profit": df["Profit"].sum(),
        "Total Orders": df["Order_ID"].nunique(),
        "Avg Order Value": df["Total_Sale"].mean(),
        "Avg Profit": df["Profit"].mean(),
        "Return Rate": df["Returned"].mean() * 100 if "Returned" in df else 0,
        "Avg Rating": df["Customer_Rating"].mean() if "Customer_Rating" in df else 0,
        "Avg Delivery Days": df["Delivery_Days"].mean(),
    }


def monthly_sales_profit(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["Order_Date"]).copy()
    d["Period"] = d["Order_Date"].dt.to_period("M").dt.to_timestamp()
    return d.groupby("Period").agg(Sales=("Total_Sale", "sum"),
                                   Profit=("Profit", "sum")).reset_index()


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Product_Category")
        .agg(Sales=("Total_Sale", "sum"), Profit=("Profit", "sum"),
             Orders=("Order_ID", "nunique"))
        .reset_index()
        .sort_values("Sales", ascending=False)
    )


def top_products(df, by="Total_Sale", n=10):
    return (
        df.groupby("Product_Name")[by].sum()
        .sort_values(ascending=False).head(n).reset_index()
    )


def region_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Region")
        .agg(Sales=("Total_Sale", "sum"), Profit=("Profit", "sum"))
        .reset_index()
    )


def salesperson_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Salesperson")
        .agg(Sales=("Total_Sale", "sum"), Profit=("Profit", "sum"),
             Orders=("Order_ID", "nunique"))
        .reset_index().sort_values("Profit", ascending=False)
    )