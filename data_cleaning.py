from datetime import date, datetime
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_FILE = Path("data.xlsx")
OUTPUT_FILE = Path("data_cleaned.xlsx")

SHEET_NAMES = {
    "orders": "Orders_Raw",
    "customers": "Customers",
    "products": "Products",
    "stores": "Stores",
    "returns": "Returns_Raw",
}

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {INPUT_FILE.resolve()}."
    )

raw = {
    key: pd.read_excel(INPUT_FILE, sheet_name=sheet_name)
    for key, sheet_name in SHEET_NAMES.items()
}

rows_before = {
    key: int(df.dropna(how="all").shape[0])
    for key, df in raw.items()
}

issues = []
logged_issue_keys = set()


def log_issue(table, record_id, column, original_value, issue, action):
    record_id = "(unknown ID)" if pd.isna(record_id) else str(record_id)
    key = (table, record_id, column)

    if key in logged_issue_keys:
        return
    logged_issue_keys.add(key)

    if pd.isna(original_value):
        original_value = ""
    else:
        original_value = str(original_value)

    issues.append(
        {
            "Table": table,
            "Record_ID": record_id,
            "Column": column,
            "Original_Value": original_value,
            "Issue": issue,
            "Action_Taken": action,
        }
    )


def tidy_text(df):
    df = df.copy().dropna(how="all")
    missing_tokens = {"", "n/a", "na", "nan", "null", "none"}

    for col in df.columns:
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            def clean_value(value):
                if isinstance(value, str):
                    value = value.strip()
                    if value.casefold() in missing_tokens:
                        return np.nan
                return value

            df[col] = df[col].apply(clean_value)

    return df


def parse_excel_date(value):
    if pd.isna(value):
        return pd.NaT

    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.to_datetime(value, errors="coerce")

    if isinstance(value, Real) and not isinstance(value, bool):
        return pd.to_datetime(
            value, unit="D", origin="1899-12-30", errors="coerce"
        )

    return pd.to_datetime(str(value).strip(), errors="coerce")


def parse_date_column(series):
    return series.apply(parse_excel_date)


def remove_duplicates(df, table_name, key_column):
    df = df.copy()

    duplicate_rows = df.duplicated(keep="first")
    for idx in df.index[duplicate_rows]:
        log_issue(
            table_name,
            df.loc[idx, key_column] if key_column in df.columns else idx,
            "Entire row",
            "Duplicate row",
            "Exact duplicate record",
            "Removed duplicate row",
        )
    df = df.drop_duplicates().copy()

    duplicate_keys = df[key_column].notna() & df.duplicated(key_column, keep="first")
    for idx in df.index[duplicate_keys]:
        log_issue(
            table_name,
            df.loc[idx, key_column],
            key_column,
            df.loc[idx, key_column],
            "Duplicate ID remains after exact duplicates were removed",
            "Kept the first record for this ID; review source data if records differ",
        )
    df = df.drop_duplicates(subset=[key_column], keep="first").copy()
    return df


orders = remove_duplicates(tidy_text(raw["orders"]), "Orders_Raw", "Order_Line_ID")
customers = remove_duplicates(tidy_text(raw["customers"]), "Customers", "Customer_ID")
products = remove_duplicates(tidy_text(raw["products"]), "Products", "Product_ID")
stores = remove_duplicates(tidy_text(raw["stores"]), "Stores", "Store_ID")
returns = remove_duplicates(tidy_text(raw["returns"]), "Returns_Raw", "Return_ID")


for col in ["Standard_Unit_Price", "Unit_Cost"]:
    products[col] = pd.to_numeric(products[col], errors="coerce")
    invalid = products[col].notna() & (products[col] <= 0)
    for idx in products.index[invalid]:
        log_issue(
            "Products",
            products.loc[idx, "Product_ID"],
            col,
            products.loc[idx, col],
            "Price or cost is not positive",
            "Set invalid value to missing for review",
        )
    products.loc[invalid, col] = np.nan

customers["Signup_Date"] = parse_date_column(customers["Signup_Date"])


orders["Order_Date"] = parse_date_column(orders["Order_Date"])

channel_map = {
    "online": "Online",
    "in-store": "In-Store",
    "b2b": "B2B",
}
channel_normalized = orders["Sales_Channel"].astype("string").str.casefold().map(channel_map)
channel_unknown = orders["Sales_Channel"].notna() & channel_normalized.isna()
for idx in orders.index[channel_unknown]:
    log_issue(
        "Orders_Raw",
        orders.loc[idx, "Order_Line_ID"],
        "Sales_Channel",
        orders.loc[idx, "Sales_Channel"],
        "Unrecognized sales channel",
        "Kept original value for manual review",
    )
orders.loc[channel_normalized.notna(), "Sales_Channel"] = channel_normalized[channel_normalized.notna()]

for col in ["Quantity", "Unit_Price", "Discount_Pct"]:
    orders[col] = pd.to_numeric(orders[col], errors="coerce")

invalid_quantity = orders["Quantity"].notna() & (
    (orders["Quantity"] <= 0) | (orders["Quantity"] % 1 != 0)
)
for idx in orders.index[invalid_quantity]:
    log_issue(
        "Orders_Raw",
        orders.loc[idx, "Order_Line_ID"],
        "Quantity",
        orders.loc[idx, "Quantity"],
        "Quantity is not a positive whole number",
        "Set invalid value to missing for review",
    )
orders.loc[invalid_quantity, "Quantity"] = np.nan
orders["Quantity"] = orders["Quantity"].astype("Int64")

invalid_discount = orders["Discount_Pct"].notna() & ~orders["Discount_Pct"].between(0, 1)
for idx in orders.index[invalid_discount]:
    log_issue(
        "Orders_Raw",
        orders.loc[idx, "Order_Line_ID"],
        "Discount_Pct",
        orders.loc[idx, "Discount_Pct"],
        "Discount is outside the valid 0 to 1 range",
        "Set invalid discount to missing for review",
    )
orders.loc[invalid_discount, "Discount_Pct"] = np.nan

standard_price = orders["Product_ID"].map(
    products.set_index("Product_ID")["Standard_Unit_Price"]
)
price_needs_repair = standard_price.notna() & (
    orders["Unit_Price"].isna()
    | ((orders["Unit_Price"] - standard_price).abs() > 0.01)
)
for idx in orders.index[price_needs_repair]:
    old_price = orders.loc[idx, "Unit_Price"]
    new_price = standard_price.loc[idx]
    log_issue(
        "Orders_Raw",
        orders.loc[idx, "Order_Line_ID"],
        "Unit_Price",
        old_price,
        "Missing or does not match the product master price",
        f"Set to Products.Standard_Unit_Price ({new_price:g})",
    )
orders.loc[price_needs_repair, "Unit_Price"] = standard_price[price_needs_repair]

known_order_date = orders.groupby("Order_ID")["Order_Date"].transform(
    lambda values: values.dropna().mode().iloc[0] if not values.dropna().empty else pd.NaT
)
fillable_date = orders["Order_Date"].isna() & known_order_date.notna()
for idx in orders.index[fillable_date]:
    log_issue(
        "Orders_Raw",
        orders.loc[idx, "Order_Line_ID"],
        "Order_Date",
        "",
        "Order date was missing",
        f"Filled from another line in the same order ({known_order_date.loc[idx].date()})",
    )
orders.loc[fillable_date, "Order_Date"] = known_order_date[fillable_date]

for col in ["Order_Date", "Quantity", "Unit_Price", "Discount_Pct"]:
    for idx in orders.index[orders[col].isna()]:
        log_issue(
            "Orders_Raw",
            orders.loc[idx, "Order_Line_ID"],
            col,
            "",
            f"Missing {col}",
            "Left missing because it could not be inferred reliably",
        )

orders["Unit_Cost"] = orders["Product_ID"].map(
    products.set_index("Product_ID")["Unit_Cost"]
)
orders["Gross_Sales"] = orders["Quantity"].astype("Float64") * orders["Unit_Price"]
orders["Discount_Amount"] = orders["Gross_Sales"] * orders["Discount_Pct"]
orders["Net_Sales"] = orders["Gross_Sales"] - orders["Discount_Amount"]
orders["Total_Cost"] = orders["Quantity"].astype("Float64") * orders["Unit_Cost"]
orders["Gross_Profit"] = orders["Net_Sales"] - orders["Total_Cost"]


returns["Return_Date"] = parse_date_column(returns["Return_Date"])
returns["Return_Quantity"] = pd.to_numeric(returns["Return_Quantity"], errors="coerce")
returns["Refund_Amount"] = pd.to_numeric(returns["Refund_Amount"], errors="coerce")

invalid_return_quantity = returns["Return_Quantity"].notna() & (
    (returns["Return_Quantity"] <= 0) | (returns["Return_Quantity"] % 1 != 0)
)
for idx in returns.index[invalid_return_quantity]:
    log_issue(
        "Returns_Raw",
        returns.loc[idx, "Return_ID"],
        "Return_Quantity",
        returns.loc[idx, "Return_Quantity"],
        "Return quantity is not a positive whole number",
        "Set invalid value to missing for review",
    )
returns.loc[invalid_return_quantity, "Return_Quantity"] = np.nan

order_lookup = orders.set_index("Order_Line_ID")
returns["_Order_Date"] = returns["Order_Line_ID"].map(order_lookup["Order_Date"])
returns["_Order_Quantity"] = returns["Order_Line_ID"].map(order_lookup["Quantity"])
returns["_Order_Unit_Price"] = returns["Order_Line_ID"].map(order_lookup["Unit_Price"])
returns["_Order_Discount_Pct"] = returns["Order_Line_ID"].map(order_lookup["Discount_Pct"])

valid_order_lines = set(orders["Order_Line_ID"].dropna())
invalid_order_line_ref = returns["Order_Line_ID"].notna() & ~returns["Order_Line_ID"].isin(valid_order_lines)
for idx in returns.index[invalid_order_line_ref]:
    log_issue(
        "Returns_Raw",
        returns.loc[idx, "Return_ID"],
        "Order_Line_ID",
        returns.loc[idx, "Order_Line_ID"],
        "Return references an unknown order line",
        "Kept row for review; could not validate against Orders_Raw",
    )

return_before_order = (
    returns["Return_Date"].notna()
    & returns["_Order_Date"].notna()
    & (returns["Return_Date"] < returns["_Order_Date"])
)
for idx in returns.index[return_before_order]:
    log_issue(
        "Returns_Raw",
        returns.loc[idx, "Return_ID"],
        "Return_Date",
        returns.loc[idx, "Return_Date"].date(),
        "Return date is earlier than the order date",
        "Set date to missing for review; no corrected date was assumed",
    )
returns.loc[return_before_order, "Return_Date"] = pd.NaT

calculated_refund = (
    returns["Return_Quantity"]
    * returns["_Order_Unit_Price"]
    * (1 - returns["_Order_Discount_Pct"])
)
valid_refund_inputs = (
    returns["Return_Quantity"].notna()
    & returns["_Order_Quantity"].notna()
    & (returns["Return_Quantity"] <= returns["_Order_Quantity"])
    & returns["_Order_Unit_Price"].notna()
    & returns["_Order_Discount_Pct"].between(0, 1)
)
missing_refund_can_be_inferred = returns["Refund_Amount"].isna() & valid_refund_inputs
for idx in returns.index[missing_refund_can_be_inferred]:
    log_issue(
        "Returns_Raw",
        returns.loc[idx, "Return_ID"],
        "Refund_Amount",
        "",
        "Refund amount was missing",
        f"Calculated from order-line price, discount, and return quantity ({calculated_refund.loc[idx]:.2f})",
    )
returns.loc[missing_refund_can_be_inferred, "Refund_Amount"] = calculated_refund[
    missing_refund_can_be_inferred
]

refund_math_available = (
    returns["Refund_Amount"].notna()
    & returns["Return_Quantity"].notna()
    & returns["_Order_Unit_Price"].notna()
    & returns["_Order_Discount_Pct"].between(0, 1)
)
refund_mismatch = refund_math_available & (
    (returns["Refund_Amount"] - calculated_refund).abs() > 1.0
)
for idx in returns.index[refund_mismatch]:
    log_issue(
        "Returns_Raw",
        returns.loc[idx, "Return_ID"],
        "Refund_Amount",
        returns.loc[idx, "Refund_Amount"],
        "Recorded refund differs from order-line price/discount math",
        "Kept the recorded amount and flagged it for manual review",
    )

return_exceeds_order = (
    returns["Return_Quantity"].notna()
    & returns["_Order_Quantity"].notna()
    & (returns["Return_Quantity"] > returns["_Order_Quantity"])
)
for idx in returns.index[return_exceeds_order]:
    log_issue(
        "Returns_Raw",
        returns.loc[idx, "Return_ID"],
        "Return_Quantity",
        returns.loc[idx, "Return_Quantity"],
        "Return quantity exceeds the quantity purchased",
        "Set quantity to missing for review; did not cap it or delete the return",
    )
returns.loc[return_exceeds_order, "Return_Quantity"] = np.nan
returns["Return_Quantity"] = returns["Return_Quantity"].astype("Int64")

for col in ["Return_Date", "Return_Quantity", "Refund_Amount"]:
    for idx in returns.index[returns[col].isna()]:
        log_issue(
            "Returns_Raw",
            returns.loc[idx, "Return_ID"],
            col,
            "",
            f"Missing {col}",
            "Left missing because it could not be inferred reliably",
        )

returns = returns.drop(
    columns=["_Order_Date", "_Order_Quantity", "_Order_Unit_Price", "_Order_Discount_Pct"]
)



for column, valid_values, table_name in [
    ("Customer_ID", set(customers["Customer_ID"].dropna()), "Customers"),
    ("Product_ID", set(products["Product_ID"].dropna()), "Products"),
    ("Store_ID", set(stores["Store_ID"].dropna()), "Stores"),
]:
    invalid_refs = orders[column].notna() & ~orders[column].isin(valid_values)
    for idx in orders.index[invalid_refs]:
        log_issue(
            "Orders_Raw",
            orders.loc[idx, "Order_Line_ID"],
            column,
            orders.loc[idx, column],
            f"Reference does not exist in {table_name}",
            "Kept row for review; reference was not guessed or replaced",
        )

cleaned_tables = {
    "Orders_Clean": orders,
    "Customers_Clean": customers,
    "Products_Clean": products,
    "Stores_Clean": stores,
    "Returns_Clean": returns,
}

summary_rows = []
for table_name, df in cleaned_tables.items():
    source_key = {
        "Orders_Clean": "orders",
        "Customers_Clean": "customers",
        "Products_Clean": "products",
        "Stores_Clean": "stores",
        "Returns_Clean": "returns",
    }[table_name]
    summary_rows.append(
        {
            "Sheet": table_name,
            "Rows_Before_Cleaning": rows_before[source_key],
            "Rows_After_Cleaning": len(df),
            "Rows_Removed": rows_before[source_key] - len(df),
            "Missing_Cells_After_Cleaning": int(df.isna().sum().sum()),
        }
    )

cleaning_summary = pd.DataFrame(summary_rows)
data_quality_issues = pd.DataFrame(
    issues,
    columns=["Table", "Record_ID", "Column", "Original_Value", "Issue", "Action_Taken"],
)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    for sheet_name, df in cleaned_tables.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    cleaning_summary.to_excel(writer, sheet_name="Cleaning_Summary", index=False)
    data_quality_issues.to_excel(writer, sheet_name="Data_Quality_Issues", index=False)

print(f"Cleaning complete. Saved workbook: {OUTPUT_FILE.resolve()}")
print("\nRows before and after cleaning:")
print(cleaning_summary.to_string(index=False))
print(f"\nData-quality items logged: {len(data_quality_issues)}")
print("Review the Data_Quality_Issues sheet for repairs and unresolved values.")
