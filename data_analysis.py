from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

# Files and settings

INPUT_FILES = Path("data_cleaned.xlsx")
OUTPUT_FILES = Path("data_analysis_outputs.xlsx")

SHEETS = {
    "orders": "Orders_Clean",
    "customers": "Customers_Clean",
    "products": "Products_Clean",
    "stores": "Stores_Clean",
    "returns": "Returns_Clean",
}

TOP_PRODUCT_PAIRS = 100

# Load and validate the cleaned data

def require_columns(df, required, table_name):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {missing}."
        )

def load_data():
    if not INPUT_FILES.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILES.resolve()}."
        )

    data = {}
    for key, sheet in SHEETS.items():
        try:
            data[key] = pd.read_excel(INPUT_FILES, sheet_name=sheet)
        except ValueError as exc:
            raise ValueError(
                f"Could not read sheet '{sheet}' from {INPUT_FILES.name}."
            ) from exc

    try:
        data["cleaning_issues"] = pd.read_excel(INPUT_FILES, sheet_name="Data_Quality_Issues")
    except ValueError:
        data["cleaning_issues"] = pd.DataFrame()

    require_columns(
        data["orders"],
        [
            "Order_Line_ID", "Order_ID", "Order_Date", "Customer_ID", "Product_ID", "Store_ID", "Sales_Channel", "Quantity", "Unit_Price", "Discount_Pct",
        ],
        SHEETS["orders"],
    )
    require_columns(
        data["customers"],
        [
            "Customer_ID", "Customer_Name", "Customer_Segment", "Loyalty_Tier", "Signup_Date", "City"
        ],
        SHEETS["customers"],
    )
    require_columns(
        data["products"],
        [
            "Product_ID", "Product_Name", "Category", "Brand", "Standard_Unit_Price", "Unit_Cost"
        ],
        SHEETS["products"],
    )
    require_columns(
        data["stores"],
        [
            "Store_ID", "Store_Name", "City", "Region", "Store_Type"
        ],
        SHEETS["stores"],
    )
    require_columns(
        data["returns"],
        [
            "Return_ID", "Order_Line_ID", "Return_Date", "Return_Quantity", "Refund_Amount", "Return_Reason"
        ],
        SHEETS["returns"],
    )

    return data

data = load_data()
orders = data["orders"].copy()
customers = data["customers"].copy()
products = data["products"].copy()
stores = data["stores"].copy()
returns = data["returns"].copy()

# Normalize types and re-calculate the main sales metrics

orders["Order_Date"] = pd.to_datetime(orders["Order_Date"], errors="coerce")
customers["Signup_Date"] = pd.to_datetime(customers["Signup_Date"], errors="coerce")
returns["Return_Date"] = pd.to_datetime(returns["Return_Date"], errors="coerce")

numeric_columns = ["Quantity", "Unit_Price", "Discount_Pct"]
for column in numeric_columns:
    orders[column] = pd.to_numeric(orders[column], errors="coerce")

products["Standard_Unit_Price"] = pd.to_numeric(products["Standard_Unit_Price"], errors="coerce")
products["Unit_Cost"] = pd.to_numeric(products["Unit_Cost"], errors="coerce")
returns["Return_Quantity"] = pd.to_numeric(returns["Return_Quantity"], errors="coerce")
returns["Refund_Amount"] = pd.to_numeric(returns["Refund_Amount"], errors="coerce")

orders["Gross_Sales"] = orders["Quantity"] * orders["Unit_Price"]
orders["Discount_Amount"] = orders["Gross_Sales"] * orders["Discount_Pct"]
orders["Net_Sales"] = orders["Gross_Sales"] - orders["Discount_Amount"]

product_cost = products.set_index("Product_ID")["Unit_Cost"]
orders["Unit_Cost_Analysis"] = orders["Product_ID"].map(product_cost)
orders["Total_Cost"] = orders["Quantity"] * orders["Unit_Cost_Analysis"]
orders["Gross_Profit"] = orders["Net_Sales"] - orders["Total_Cost"]
orders["Gross_Margin_Pct"] = np.where(
    orders["Net_Sales"].notna() & (orders["Net_Sales"] != 0),
    orders["Gross_Profit"] / orders["Net_Sales"],
    np.nan,
)

# Build a star-schema-like analytics table

customer_dim = customers.rename(columns={"City": "Customer_City"}).copy()
store_dim = stores.rename(columns={"City": "Store_City"}).copy()
product_dim = products[
    ["Product_ID", "Product_Name", "Category", "Brand", "Standard_Unit_Price"]
].copy()

orders_enriched = orders.merge(
    customer_dim,
    on="Customer_ID",
    how="left",
    validate="many_to_one",
    suffixes=("", "_Customer"),
)
orders_enriched = orders_enriched.merge(
    product_dim,
    on="Product_ID",
    how="left",
    validate="many_to_one",
)
orders_enriched = orders_enriched.merge(
    store_dim,
    on="Store_ID",
    how="left",
    validate="many_to_one"
)

orders_enriched["Year"] = orders_enriched["Order_Date"].dt.year.astype("Int64")
orders_enriched["Quarter"] = orders_enriched["Order_Date"].dt.to_period("Q").astype("string")
orders_enriched["Year_Month"] = orders_enriched["Order_Date"].dt.to_period("M").astype("string")
orders_enriched["Day_of_Week"] = orders_enriched["Order_Date"].dt.day_name()
orders_enriched["Weekday_Number"] = orders_enriched["Order_Date"].dt.dayofweek
orders_enriched["Month_Number"] = orders_enriched["Order_Date"].dt.month.astype("Int64")
orders_enriched["Month_Name"] = orders_enriched["Order_Date"].dt.month_name()
orders_enriched["Signup_Month"] = orders_enriched["Signup_Date"].dt.to_period("M").astype("string")

if returns.empty:
    return_by_line = pd.DataFrame(
        columns=["Order_Line_ID", "Return_Record_Count", "Returned_Units", "Refund_Amount"]
    )
else:
    return_by_line = (
        returns.groupby("Order_Line_ID", dropna=False)
        .agg(
            Return_Record_Count=("Return_ID", "nunique"),
            Returned_Units=("Return_Quantity", "sum"),
            Refund_Amount=("Refund_Amount", "sum"),
        ).reset_index()
    )

orders_enriched = orders_enriched.merge(
    return_by_line,
    on="Order_Line_ID",
    how="left",
    validate="one_to_one",
)
for column in ["Return_Record_Count", "Returned_Units", "Refund_Amount"]:
    orders_enriched[column] = orders_enriched.fillna(0)

orders_enriched["Line_Return_Flag"] = orders_enriched["Returned_Units"] > 0
orders_enriched["Sales_After_Recorded_Refunds"] = (
    orders_enriched["Net_Sales"] - orders_enriched["Refund_Amount"]
)
orders_enriched["Return_Unit_Rate"] = np.where(
    orders_enriched["Quantity"].notna() & (orders_enriched["Quantity"] > 0),
    orders_enriched["Returned_Units"] / orders_enriched["Quantity"],
    np.nan,
)

# Common summary functions

def safe_divide(numerator, denominator):
    if pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator

def summarize_sales(frame, group_columns):
    if isinstance(group_columns, str):
        group_columns = [group_columns]

    result = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            Order_Count=("Order_ID", "nunique"),
            Order_Line_Count=("Order_Line_ID", "nunique"),
            Customer_Count=("Customer_ID", "nunique"),
            Units_Sold=("Quantity", "sum"),
            Gross_Sales=("Gross_Sales", "sum"),
            Discount_Amount=("Discount_Amount", "sum"),
            Net_Sales=("Net_Sales", "sum"),
            Total_Sales=("Total_Cost", "sum"),
            Gross_Profit=("Gross_Profit", "sum"),
            Average_Discount_Pct=("Discount_Pct", "mean"),
            Return_Records=("Return_Record_Count", "sum"),
            Returned_Order_Lines=("Line_Return_Flag", "sum"),
            Returned_Units=("Returned_Units", "sum"),
            Refund_Amount=("Refund_Amount", "sum"),
        ).reset_index()
    )
    result["Gross_Margin_Pct"] = np.where(
        result["Net_Sales"] != 0,
        result["Gross_Profit"] / result["Net_Sales"],
        np.nan,
    )
    result["Discount_Rate_of_Gross_Sales"] = np.where(
        result["Gross_Sales"] != 0,
        result["Discount_Amount"] / result["Gross_Sales"],
        np.nan,
    )
    result["Average_Order_Value"] = np.where(
        result["Order_Count"] != 0,
        result["Net_Sales"] / result["Order_Count"],
        np.nan,
    )
    result["Sales_After_Recorded_Refunds"] = result["Net_Sales"] - result["Refund_Amount"]
    result["Return_Rate_by_Units"] = np.where(
        result["Units_Sold"] > 0, result["Returned_Units"] / result["Units_Sold"], np.nan
    )
    result["Return_Line_Rate"] = np.where(
        result["Order_Line_Count"] > 0, result["Returned_Order_Lines"] / result["Order_Line_Count"], np.nan
    )
    return result

def trend_summary(frame, period_column):
    result = summarize_sales(frame.dropna(subset=[period_column]), period_column)
    result = result.sort_value(period_column).reset_index(drop=True)
    result["Sales_Period_Growth_Pct"] = result["Net_Sales"].pct_change()
    result["Profit_Period_Growth_Pct"] = result["Gross_Profit"].pct_change()
    result["Sales_3_Period_Rolling_Average"] = result["Net_Sales"].rolling(3, min_periods=1)
    result[["Sales_Period_Growth_Pct", "Profit_Period_Growth_Pct"]] = result[
        ["Sales_Period_Growth_Pct", "Profit_Period_Growth_Pct"]
    ].replace([np.inf, -np.inf], np.nan)
    return result

def metric_row(name, value, definition):
    return {"Metric": name, "Value": value, "Definition": definition}

# Executive summary and core sales KPIs

valid_sales_dates = orders_enriched["Order_Date"].dropna()
first_order_date = valid_sales_dates.min() if not valid_sales_dates.empty else pd.NaT
last_order_date = valid_sales_dates.max() if not valid_sales_dates.empty else pd.NaT
net_sales_total = orders_enriched["Net_Sales"].sum(min_count=1)
gross_sales_total = orders_enriched["Gross_Sales"].sum(min_count=1)
discount_total = orders_enriched["Discount_Amount"].sum(min_count=1)
cost_total = orders_enriched["Total_Cost"].sum(min_count=1)
profit_total = orders_enriched["Gross_Profit"].sum(min_count=1)
order_total = orders_enriched["Order_ID"].nunique()
customer_total = orders_enriched["Customer_ID"].nunique()
units_total = orders_enriched["Quantity"].sum(min_count=1)
return_units_total = orders_enriched["Return_Quantity"].sum(min_count=1) if not returns.empty else 0
refund_total = orders_enriched["Refund_Amount"].sum(min_count=1) if not returns.empty else 0
returned_order_lines = returns.loc[returns["Return_Quantity"].fillna(0) > 0, "Order_Line_ID"].nunique()

executive_summary = pd.DataFrame(
    [
        metric_row("First Order Date", first_order_date, "Earliest valid Order_Date in the orders data."),
        metric_row("Last Order Date", last_order_date, "Latest valid Order_Date in the orders data."),
        metric_row("Days Covered", (last_order_date - first_order_date).days if pd.notna(first_order_date) and pd.notna(last_order_date) else np.nan, "Calendar days between first and last valid order date."),
        metric_row("Gross Sales", gross_sales_total, "Quantity × Unit_Price before discounts; excludes tax/shipping because not provided."),
        metric_row("Discount Amount", discount_total, "Gross sales × Discount_Pct."),
        metric_row("Net Sales", net_sales_total, "Sales after discounts, before refunds."),
        metric_row("Total Cost", cost_total, "Quantity × Unit_Cost from the product master."),
        metric_row("Gross Profit", profit_total, "Net Sales − estimated product cost; not net income."),
        metric_row("Gross Margin %", safe_divide(profit_total, net_sales_total), "Gross Profit ÷ Net Sales."),
        metric_row("Discount % of Gross Sales", safe_divide(discount_total, gross_sales_total), "Discount Amount ÷ Gross Sales."),
        metric_row("Orders", order_total, "Distinct Order_ID count."),
        metric_row("Order Lines", orders_enriched["Order_Line_ID"].nunique(), "Distinct Order_Line_ID count."),
        metric_row("Units Sold", units_total, "Sum of known order quantities; missing quantities are excluded."),
        metric_row("Customers With Orders", customer_total, "Distinct non-missing Customer_ID values in orders."),
        metric_row("Average Order Value", safe_divide(net_sales_total, order_total), "Net Sales ÷ distinct orders."),
        metric_row("Average Sales per Customer", safe_divide(net_sales_total, customer_total), "Net Sales ÷ customers with orders."),
        metric_row("Average Units per Order", safe_divide(units_total, order_total), "Known units sold ÷ distinct orders."),
        metric_row("Average Selling Price per Unit", safe_divide(net_sales_total, units_total), "Net Sales ÷ known units sold."),
        metric_row("Average Line Discount %", orders_enriched["Discount_Pct"].mean(), "Arithmetic average discount percentage per order line; not weighted by sales."),
        metric_row("Return Records", returns["Return_ID"].nunique(), "Distinct Return_ID count."),
        metric_row("Returned Units", return_units_total, "Sum of known Return_Quantity values."),
        metric_row("Refund Amount", refund_total, "Sum of recorded/calculated Refund_Amount values; see return quality checks."),
        metric_row("Return Rate by Units %", safe_divide(return_units_total, units_total), "Known returned units ÷ known units sold; interpret with missing return quantities in mind."),
        metric_row("Order Lines With a Return", returned_order_lines, "Distinct order lines with a positive known return quantity."),
        metric_row("Return Line Rate %", safe_divide(returned_order_lines, orders_enriched["Order_Line_ID"].nunique()), "Order lines with known positive returns ÷ all order lines."),
        metric_row("Sales After Recorded Refunds", net_sales_total - refund_total if pd.notna(net_sales_total) else np.nan, "Net Sales minus refunds; useful as a simplified indicator, not a full accounting measure."),
    ]
)

# Sales trends: daily, monthly, quarterly, yearly, weekday

monthly_sales = trend_summary(orders_enriched, "Year_Month")
quarterly_sales = trend_summary(orders_enriched, "Quarter")
yearly_sales = trend_summary(orders_enriched, "Year")
daily_sales = summarize_sales(orders_enriched.dropna(subset=["Order_Date"]), "Order_Date")
daily_sales = daily_sales.sort_values("Order_Date").reset_index(drop=True)

weekly_sales = summarize_sales(
    orders_enriched(subset=["Day_of_Week"]), ["Weekday_Number", "Day_of_Week"]
).sort_values("Weekday_Number")

daily_by_weekday = (
    orders_enriched.dropna(subset=["Order_Date", "Day_of_Week"])
    .groupby(["Order_Date", "Weekday_Number", "Day_of_Week"], dropna=False)
    .agg(Daily_Net_Sales=("Net_Sales", "sum"), Daily_Gross_Profit=("Gross_Profit", "sum"))
    .reset_index()
)
average_daily_sales_by_weekday = (
    daily_by_weekday.groupby(["Weekday_Number", "Day_of_Week"], dropna=False)
    .agg(Trading_Day=("Order_Date", "nunique"), Average_Daily_Net_Sales=("Daily_Net_Sales", "mean"),
         Median_Daily_Net_Sales=("Daily_Net_Sales", "median"), Average_Daily_Gross_Profit=("Daily_Gross_Profit", "mean"))
    .reset_index().sort_values("Weekday_Number")
)

month_seasonality = summarize_sales(orders_enriched.dropna(subset=["Month_Number"]), ["Month_Number", "Month_Name"]).sort_values("Month_Number")

monthly_weekday = summarize_sales(
    orders_enriched.dropna(subset=["Year_Month", "Day_of_Week"]),
    ["Year_Month", "Day_of_Week"],
)
monthly_channel_sales = summarize_sales(orders_enriched.dropna(subset=["Year_Month"]), ["Year_Month", "Sales_Channel"])
monthly_category_sales = summarize_sales(orders_enriched.dropna(subset=["Year_Month"]), ["Year_Month", "Category"])
monthly_store_sales = summarize_sales(orders_enriched.dropna(subset=["Year_Month"]), ["Year_Month", "Store_ID", "Store_Name", "Region"])

# Dimension performance: channel, store, region, products, customers

channel_performance = summarize_sales(orders_enriched, "Sales_Channel").sort_values("Net_Sales", ascending=False)
region_performance = summarize_sales(orders_enriched, "Region").sort_values("Net_Sales", asceding=False)
store_performance = summarize_sales(
    orders_enriched,
    ["Store_ID", "Store_Name", "Store_City", "Region", "Store_Type"]
).sort_values("Net_Sales", ascending=False)
store_type_performance = summarize_sales(orders_enriched, "Store_Type").sort_values("Net_Sales", ascending=False)
category_performance = summarize_sales(orders_enriched, "Category").sort_values("Net_Sales", ascending=False)
brand_performance = summarize_sales(orders_enriched, "Brand").sort_value("Net_Sales", ascending=False)
product_performance = summarize_sales(
    orders_enriched,
    ["Product_ID", "Product_Name", "Category", "Brand"]
).sort_values("Net_Sales", ascending=False)
customer_segment_performance = summarize_sales(orders_enriched, "Customer_Segment").sort_values("Net_Sales", ascending=False)
loyalty_tier_performance = summarize_sales(orders_enriched, "Loyalty_Tier").sort_values("Net_Sales", ascending=False)
customer_city_performance = summarize_sales(orders_enriched, "Customer_City").sort_values("Net_Sales", ascending=False)

channel_category_pivot = pd.pivot_table(
    orders_enriched,
    index="Sales_Channel",
    columns="Category",
    values="Net_Sales",
    aggfunc="sum",
    fill_value=0,
    dropna=False,
).reset_index()
region_category_pivot = pd.pivot_table(
    orders_enriched,
    index="Region",
    columns="Category",
    values="Net_Sales",
    aggfunc="sum",
    fill_value=0,
    dropna=False,
).reset_index()
segment_channel_pivot = pd.pivot_table(
    orders_enriched,
    index="Customer_Segment",
    columns="Sales_Channel",
    values="Net_Sales",
    aggfunc="sum",
    fill_value=0,
    dropna=False
).reset_index()

# Discount and profitability analysis

def discount_band(value):
    if pd.isna(value):
        return "Unknown"
    if value == 0:
        return "No Discount"
    if value <= 0.10:
        return ">0-10%"
    if value <= 0.20:
        return ">10-20%"
    if value <= 0.30:
        return ">20-30%"
    return ">30%"

orders_enriched["Discount_Band"] = orders_enriched["Discount_Pct"].apply(discount_band)
discount_order = ["No Discount", ">0-10%", ">10-20%", ">20-30%", ">30%", "Unknown"]
discount_analysis = summarize_sales(orders_enriched, "Discount_Band")
discount_analysis["Discount_Band"] = pd.Categorical(
    discount_analysis["Discount_Band"], categories=discount_order, ordered=True
)
discount_analysis = discount_analysis.sort_values("Discount_Band").reset_index(drop=True)

discount_line_stats = (
    orders_enriched.groupby("Discount_Band", dropna=False)
    .agg(
        Order_Line_Count=("Order_Line_ID", "nunique"),
        Quantity_Median=("Quantity", "median"),
        Net_Sales_Median=("Net_Sales", "median"),
        Gross_Profit_Median=("Gross_Profit", "median"),
        Gross_Margin_Median=("Gross_Margin_Pct", "median"),
        Discount_Pct_Mean=("Discount_Pct", "mean"),
    ).reset_index()
)

# Customer behavior: RFM, repeat buying and signup cohorts

order_level = (
    orders_enriched.groupby("Order_ID", dropna=False)
    .agg(
        Order_Date=("Order_Date", "min"),
        Customer_ID=("Customer_ID", "first"),
        Sales_Channel=("Sales_Channel", "first"),
        Store_ID=("Store_ID", "first"),
        Order_Net_Sales=("Net_Sales", "sum"),
        Order_Gross_Profit=("Gross_Profit", "sum"),
        Order_Gross_Sales=("Gross_Sales", "sum"),
        Order_Discount_Amount=("Discount_Amount", "sum"),
        Order_Units=("Quantity", "sum"),
        Order_Line_Count=("Order_Line_ID", "nunique"),
        Distinct_Products=("Product_ID", "nunique"),
    ).reset_index()
)
order_level["Year_Month"] = order_level["Order_Date"].dt.to_period("M").astype("string")
order_level["Average_Line_Value"] = np.where(
    order_level["Order_Line_Count"] > 0,
    order_level["Order_Net_Sales"] / order_level["Order_Line_Count"],
    np.nan,
)

snapshot_date = last_order_date
customer_orders = order_level.dropna(subset=["Customer_ID"]).copy()
if pd.notna(snapshot_date):
    rfm = (
        customer_orders.groupby("Customer_ID")
        .agg(
            Last_Order_Date=("Order_Date", "max"),
            Frequency_Orders=("Order_ID", "nunique"),
            Monetary_Net_Sales=("Order_Net_Sales", "sum"),
        ).reset_index()
    )
    rfm["Recency_Days"] = (snapshot_date - rfm["Last_Order_Date"]).dt.days
else:
    rfm = pd.DataFrame(columns=["Customer", "Last_Order_Date", "Frequency_Orders", "Monetary_Net_Sales", "Recency_Days"])

rfm["Frequency_Orders"] = rfm["Frequency_Orders"].fillna(0)
rfm["Monetary_Net_Sales"] = rfm["Monetary_Net_Sales"].fillna(0)

active_rfm = rfm[rfm["Frequency_Orders"] > 0].copy()
if not active_rfm.empty:
    active_rfm["R_Score"] = (6 - np.ceil(active_rfm["Recency_Days"].rank(method="average", pct=True) * 5)).clip(1, 5)
    active_rfm["F_Score"] = np.ceil(active_rfm["Frequency_Orders"].rank(method="average", pct=True) * 5).clip(1, 5)
    active_rfm["M_Score"] = np.ceil(active_rfm["Monetary_Net_Sales"].rank(method="average", pct=True) * 5).clip(1, 5)
    for score_column in ["R_Score", "F_Score", "M_Score"]:
        active_rfm[score_column] = active_rfm[score_column].astype("Int64")
    active_rfm["RFM_Score"] = (
        active_rfm["R_Score"].astype(str)
        + active_rfm["F_Score"].astype(str)
        + active_rfm["M_Score"].astype(str)
    )
    active_rfm["RFM_Total"] = active_rfm[["R_Score", "F_Score", "M_Score"]].sum(axis=1)
    active_rfm["Customer_Group"] = np.select(
        [
            (active_rfm["R_Score"] >= 4) & (active_rfm["F_Score"] >= 4) & (active_rfm["M_Score"] >= 4),
            (active_rfm["R_Score"] >= 3) & (active_rfm["F_Score"] >= 3),
            (active_rfm["R_Score"] <= 2) & (active_rfm["F_Score"] >= 3),
            (active_rfm["R_Score"] <= 2) & (active_rfm["F_Score"] <= 2),
        ],
        ["High Value / Active", "Loyal / Regular", "At Risk / Previously Frequent", "Inactive / Low Frequency"],
        default="Needs Review",
    )
else:
    active_rfm["R_Score"] = pd.Series(dtype="Int64")
    active_rfm["F_Score"] = pd.Series(dtype="Int64")
    active_rfm["M_Score"] = pd.Series(dtype="Int64")
    active_rfm["RFM_Score"] = pd.Series(dtype="string")
    active_rfm["RFM_Total"] = pd.Series(dtype="float")
    active_rfm["Customer_Group"] = pd.Series(dtype="string")

rfm = rfm.merge(
    active_rfm[["Customer_ID", "R_Score", "F_Score", "M_Score", "RFM_Score", "RFM_Total", "Customer_Group"]],
    on="Customer_ID", how="left", validate="one_to_one",
)
rfm["Customer_Group"] = rfm["Customer_Group"].fillna("No Recorded Orders")

customer_purchase_summary = (
    customer_orders.groupby("Customer_ID")
    .agg(
        Order_Count=("Order_ID", "nunique"),
        Net_Sales=("Order_Net_Sales", "sum"),
        Gross_Profit=("Order_Gross_Profit", "sum"),
        First_Order_Date=("Order_Date", "min"),
        Last_Order_Date=("Order_Date", "max"),
    ).reset_index()
)
customer_purchase_summary["Repeat_Customer_Flag"] = customer_purchase_summary["Order_Count"] >= 2
customer_purchase_summary["Customer_Type"] = np.where(
    customer_purchase_summary["Repeat_Customer_Flag"], "Repeat Customer", "One-Time Customer"
)
repeat_purchase_summary = (
    customer_purchase_summary.groupby("Customer_Type")
    .agg(
        Customer_Count=("Customer_ID", "nunique"),
        Orders=("Order_Count", "sum"),
        Net_Sale=("Net_Sales", "sum"),
        Gross_Profit=("Gross_Profit", "sum"),
        Average_Orders_Per_Customer=("Order_Count", "mean"),
        Average_Sales_Per_Customer=("Net_Sales", "mean"),
    ).reset_index()
)

cohort_orders = orders_enriched.dropna(subset=["Signup_Month"]).copy()
cohort_performance = summarize_sales(cohort_orders, "Signup_Month")
cohort_sizes = (
    customers.dropna(subset=["Signup_Date"])
    .assign(Signup_Month=lambda frame: frame["Signup_Date"].dt.to_period("M").astype("string"))
    .groupby("Signup_Month")
    .agg(Registered_Customer_Count=("Customer_ID", "nunique"))
    .reset_index()
)
cohort_performance = cohort_sizes.merge(cohort_performance, on="Signup_Month", how="left")
cohort_performance["Sales_Per_Registered_Customer"] = np.where(
    cohort_performance["Registered_Customer_Count"] > 0,
    cohort_performance["Net_Sales"] / cohort_performance["Registered_Customer_Count"],
    np.nan,
)

retention_source = orders_enriched.dropna(subset=["Customer_ID", "Signup_Date", "Order_Date"]).copy()
retention_source["Cohort_Month"] = retention_source["Signup_Date"].dt.to_period("M").astype("string")
retention_source["Order_Month"] = retention_source["Order_Date"].dt.to_period("M").astype("string")
retention_source["Months_Since_Signup"] = (
    (retention_source["Order_Date"].dt.year - retention_source["Signup_Date"].dt.year) * 12
    + retention_source["Order_Date"].dt.month - retention_source["Signup_Date"].dt.month
)
retention_source = retention_source[
    (retention_source["Months_Since_Signup"] >= 0)
    & (retention_source["Order_Date"] >= retention_source["Signup_Date"])
]
customer_cohort_sizes = (
    customers.dropna(subset=["Signup_Date"])
    .assign(Cohort_Month=lambda frame: frame["Signup_Date"].dt.to_period("M").astype("string"))
    .groupby("Cohort_Month").agg(Cohort_Size=("Customer_ID", "nunique")).reset_index()
)
customer_retention_cohort = (
    retention_source.groupby(["Cohort_Month", "Order_Month", "Month_Since_Signup"], dropna=False)
    .agg(Active_Customers=("Customer_ID", "nunique"), Orders=("Order_ID", "nunique"), Net_Sales=("Net_Sales", "sum"))
    .reset_index()
    .merge(customer_cohort_sizes, on="Cohort_Month", how="left")
)
customer_retention_cohort["Customer_Retention_Pct"] = np.where(
    customer_retention_cohort["Cohort_Size"] > 0,
    customer_retention_cohort["Active_Customers"] / customer_retention_cohort["Cohort_Size"],
    np.nan,
)
if not customer_retention_cohort.empty:
    customer_retention_pivot = customer_retention_cohort.pivot_table(
        index="Cohort_Month", columns="Month_Since_Signup", values="Customer_Retention_Pct", aggfunc="mean"
    ).reset_index()
else:
    customer_retention_pivot = pd.DataFrame(columns=["Cohort_Month"])

# Order / Basket analysis and product pair affinity

basket_summary = pd.DataFrame(
    [
        metric_row("Average distinct products per order", order_level["Distinct_Products"].mean(), "Average count of distinct Product_IDs in an order."),
        metric_row("Median distinct products per order", order_level["Distinct_Products"].median(), "Median count of distinct Product_IDs in an order."),
        metric_row("Average order lines", order_level["Order_Line_Count"].mean(), "Average count of order lines per Order_ID."),
        metric_row("Average order units", order_level["Order_Units"].mean(), "Average sum of known quantity by order."),
        metric_row("Median order value", order_level["Order_Net_Sales"].median(), "Median net sales per order."),
        metric_row("90th percentile order value", order_level["Order_Net_Sales"].quantile(0.90), "90th percentile net sales per order."),
    ]
)

product_name_lookup = product_dim.set_index("Product_ID")["Product_Name"].to_dict()
product_orders = {}
pair_counts = {}
number_of_orders_for_affinity = orders_enriched["Order_ID"].nunique()
for order_id, group in orders_enriched.dropna(subset=["Product_ID"]).groupby("Order_ID"):
    product_ids = sorted(set(group["Product_ID"].astype(str)))
    for product_id in product_ids:
        product_orders[product_id] = product_orders.get(product_id, 0) + 1
    for product_a, product_b in combinations(product_ids, 2):
        pair_counts[(product_a, product_b)] = pair_counts.get((product_a, product_b), 0) + 1

pair_rows = []
for (product_a, product_b), pair_order_count in pair_counts.items():
    count_a = product_orders.get(product_a, 0)
    count_b = product_orders.get(product_b, 0)
    expected_if_independent = (count_a * count_b / number_of_orders_for_affinity) if number_of_orders_for_affinity else np.nan
    lift = (pair_order_count / expected_if_independent) if expected_if_independent and expected_if_independent > 0 else np.nan
    pair_rows.append(
        {
            "Product_A_ID": product_a,
            "Product_A_Name": product_name_lookup.get(product_a, "Unknown Product"),
            "Product_B_ID": product_b,
            "Product_B_Name": product_name_lookup.get(product_b, "Unknown Product"),
            "Orders_Containing_Both": pair_order_count,
            "Support_Pct": safe_divide(pair_order_count, number_of_orders_for_affinity),
            "Orders_With_Product_A": count_a,
            "Orders_With_Product_B": count_b,
            "Lift": lift,
        }
    )
product_pair_affinity = pd.DataFrame(pair_rows)
if not product_pair_affinity.empty:
    product_pair_affinity = product_pair_affinity.sort_values(
        ["Orders_Containing_Both", "Lift"], ascending=[False, False]
    ).head(TOP_PRODUCT_PAIRS)

# Product revenue concentration (ABC / PARETO)

product_abc = product_performance.dropna(subset=["Product_ID"]).copy().sort_values("Net_Sales", ascending=False).reset_index(drop=True)
product_abc["ABC_Class"] = np.select(
    [product_abc["Cumulative_Revenue_Share"] <= 0.80, product_abc["Cumulative_Revenue_Share"] <= 0.95],
    ["A", "B"],
    default="C",
)

# Returns: reasons, product, channel and store

if returns.empty:
    return_enriched = returns.copy()
else:
    order_return_details = orders_enriched[
        [
            "Order_Line_ID", "Order_ID", "Order_Date", "Customer_ID", "Customer_Segment",
            "Loyalty_Tier", "Product_ID", "Product_Name", "Category", "Brand",
            "Store_ID", "Store_Name", "Region", "Sales_Channel", "Quantity", "Unit_Price", "Discount_Pct", "Net_Sales",
        ]
    ].drop_duplicates("Order_Line_ID")
    return_enriched = returns.merge(
        order_return_details,
        on="Order_Line_ID",
        how="left",
        validate="many_to_one",
    )
    return_enriched["Days_From_Order_To_Return"] = (
        return_enriched["Return_Date"] - return_enriched["Order_Date"]
    ).dt.days

return_records = returns["Return_ID"].nunique()
return_summary = pd.DataFrame(
    [
        metric_row("Return Records", return_records, "Distinct Return_IDs."),
        metric_row("Returned Order Lines", returns.loc[returns["Return_Quantity"].fillna(0) > 0, "Order_Line_ID"].nunique(), "Order lines with a positive known return quantity."),
        metric_row("Returned Units", returns["Return_Quantity"].sum(min_count=1) if not returns.empty else 0, "Sum of known returned quantities."),
        metric_row("Refund Amount", returns["Refund_Amount"].sum(min_count=1) if not returns.empty else 0, "Sum of known refunds. Includes inferred values already filled during cleaning, where applicable."),
        metric_row("Average Refund per Return", safe_divide(returns["Refund_Amount"].sum(min_count=1), return_records) if not returns.empty else np.nan, "Known refund amount divided by return records."),
        metric_row("Median Days to Return", return_enriched["Days_From_Order_To_Return"].median() if "Days_From_Order_To_Return" in return_enriched.columns else np.nan, "Median return date minus original order date for matched return records."),
    ]
)

if not returns.empty:
    return_reasons = (
        return_enriched.groupby("Return_Reasons", dropna=False)
        .agg(
            Return_Records=("Return_ID", "nunique"),
            Returned_Units=("Return_Quantity", "sum"),
            Refund_Amount=("Refund_Amount", "sum"),
            Average_Days_To_Return=("Days_From_Order_To_Return", "mean"),
        )
        .reset_index()
        .sort_values("Return_Records", ascending=False)
    )
    returned_products = (
        return_enriched.groupby(["Product_ID", "Product_Name", "Category", "Brand"], dropna=False)
        .agg(
            Return_Records=("Return_ID", "nunique"),
            Returned_Units=("Return_Quantity", "sum"),
            Refund_Amount=("Refund_Amount", "sum"),
        )
        .reset_index()
        .sort_values("Return_Records", ascending=False)
    )
    returns_by_channel = (
        return_enriched.groupby("Sales_Channel", dropna=False)
        .agg(
            Return_Records=("Return_ID", "nunique"),
            Returned_Units=("Return_Quantity", "sum"),
            Refund_Amount=("Refund_Amount", "sum"),
        )
        .reset_index()
    )
    returns_by_store = (
        return_enriched.groupby(["Store_ID", "Store_Name", "Region"], dropna=False)
        .agg(
            Return_Records=("Return_ID", "nunique"),
            Returned_Units=("Return_Quantity", "sum"),
            Refund_Amount=("Refund_Amount", "sum"),
        )
        .reset_index()
    )
else:
    return_reasons = pd.DataFrame(columns=["Return_Reason", "Return_Records", "Returned_Units", "Refund_Amount", "Average_Days_To_Return"])
    returned_products = pd.DataFrame(columns=["Product_ID", "Product_Name", "Category", "Brand", "Return_Records", "Returned_Units", "Refund_Amount"])
    returns_by_channel = pd.DataFrame(columns=["Sales_Channel", "Return_Records", "Returned_Units", "Refund_Amount"])
    returns_by_store = pd.DataFrame(columns=["Store_ID", "Store_Name", "Region", "Return_Records", "Returned_Units", "Refund_Amount"])

product_return_rates = product_performance.merge(
    returned_products[["Product_ID", "Return_Records", "Returned_Units", "Refund_Amount"]],
    on="Product_ID", how="left", validate="one_to_one", suffixes=("", "_Return")
)
for column in ["Return_Records", "Returned_Units", "Refund_Amount"]:
    product_return_rates[column] = product_return_rates[column].fillna(0)
product_return_rates["Returned_Units_Rate"] = np.where(
    product_return_rates["Units_Sold"] > 0, product_return_rates["Returned_Units"] / product_return_rates["Units_Sold"], np.nan
)
product_return_rates["Return_Record_Rate_Per_Order_Line"] = np.where(
    product_return_rates["Order_Line_Count"] > 0, product_return_rates["Return_Records"] / product_return_rates["Order_Line_Count"], np.nan
)