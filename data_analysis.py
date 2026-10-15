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

def safe_devide(numerator, denominator):
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