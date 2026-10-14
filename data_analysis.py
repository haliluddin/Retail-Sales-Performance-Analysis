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