
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTION_FILE = (
    PROJECT_ROOT / "reports" / "test_forecast_predictions.csv"
)

TEST_FILE = (
    PROJECT_ROOT / "data" / "processed" / "test_model_ready.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load forecast predictions
# --------------------------------------------------

print("Loading forecast predictions...")

predictions = pd.read_csv(PREDICTION_FILE)

print("Prediction shape:", predictions.shape)
print(predictions.head())


# --------------------------------------------------
# 3. Load test dataset
# --------------------------------------------------

print("\nLoading test dataset...")

test = pd.read_csv(TEST_FILE)

# Attach store, family and date information
data = test[["id", "date", "store_nbr", "family"]].merge(
    predictions[["id", "sales"]],
    on="id",
    how="left",
    validate="one_to_one"
)

data["date"] = pd.to_datetime(data["date"])

if data["sales"].isna().any():
    raise ValueError("Some predictions are missing!")

print("Combined dataset shape:", data.shape)


# --------------------------------------------------
# 4. Product family analysis
# --------------------------------------------------

print("\nAnalyzing product family performance...")

family_sales = (
    data.groupby("family")["sales"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

family_sales.to_csv(
    REPORT_DIR / "forecast_family_analysis.csv",
    index=False
)

print("\nTop 10 product families:")
print(family_sales.head(10))


# --------------------------------------------------
# 5. Store analysis
# --------------------------------------------------

print("\nAnalyzing store performance...")

store_sales = (
    data.groupby("store_nbr")["sales"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

store_sales.to_csv(
    REPORT_DIR / "forecast_store_analysis.csv",
    index=False
)

print("\nTop 10 stores:")
print(store_sales.head(10))


# --------------------------------------------------
# 6. Daily forecast analysis
# --------------------------------------------------

print("\nAnalyzing daily forecast trends...")

daily_sales = (
    data.groupby("date")["sales"]
    .sum()
    .reset_index()
)

daily_sales.to_csv(
    REPORT_DIR / "forecast_daily_analysis.csv",
    index=False
)

print("\nDaily forecast sales:")
print(daily_sales)


# --------------------------------------------------
# 7. Product family chart
# --------------------------------------------------

plt.figure(figsize=(12, 7))

top_families = family_sales.head(10).sort_values(
    "sales",
    ascending=True
)

plt.barh(
    top_families["family"],
    top_families["sales"]
)

plt.title("Top 10 Product Families - Forecasted Sales")
plt.xlabel("Predicted Sales")
plt.ylabel("Product Family")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "forecast_top_families.png"
)

plt.close()


# --------------------------------------------------
# 8. Store chart
# --------------------------------------------------

plt.figure(figsize=(12, 7))

top_stores = store_sales.head(10).sort_values(
    "sales",
    ascending=True
)

plt.barh(
    top_stores["store_nbr"].astype(str),
    top_stores["sales"]
)

plt.title("Top 10 Stores - Forecasted Sales")
plt.xlabel("Predicted Sales")
plt.ylabel("Store Number")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "forecast_top_stores.png"
)

plt.close()


# --------------------------------------------------
# 9. Daily forecast chart
# --------------------------------------------------

plt.figure(figsize=(13, 6))

plt.plot(
    daily_sales["date"],
    daily_sales["sales"],
    marker="o"
)

plt.title("Daily Forecasted Sales - August 2017")
plt.xlabel("Date")
plt.ylabel("Predicted Sales")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "forecast_daily_trend.png"
)

plt.close()


# --------------------------------------------------
# 10. Generate business summary
# --------------------------------------------------

summary = {
    "forecast_records": int(len(data)),
    "forecast_days": int(data["date"].nunique()),
    "total_predicted_sales": round(
        float(data["sales"].sum()), 2
    ),
    "average_predicted_sales_per_record": round(
        float(data["sales"].mean()), 2
    ),
    "top_product_family": str(
        family_sales.iloc[0]["family"]
    ),
    "top_product_family_sales": round(
        float(family_sales.iloc[0]["sales"]), 2
    ),
    "top_store": int(
        store_sales.iloc[0]["store_nbr"]
    ),
    "top_store_sales": round(
        float(store_sales.iloc[0]["sales"]), 2
    ),
    "highest_forecast_day": str(
        daily_sales.loc[
            daily_sales["sales"].idxmax(), "date"
        ].date()
    ),
    "highest_daily_predicted_sales": round(
        float(daily_sales["sales"].max()), 2
    )
}

with open(
    REPORT_DIR / "forecast_business_summary.json",
    "w"
) as file:
    json.dump(summary, file, indent=4)


# --------------------------------------------------
# 11. Final output
# --------------------------------------------------

print("\nFORECAST BUSINESS SUMMARY")
print("------------------------")

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nForecast analysis completed successfully!")
