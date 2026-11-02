import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Locate project folders
project_root = Path(__file__).resolve().parents[1]

data_folder = project_root / "data" / "processed"
figure_folder = project_root / "reports" / "figures"

figure_folder.mkdir(parents=True, exist_ok=True)

# 2. Load model-ready training data
df = pd.read_csv(
    data_folder / "train_model_ready.csv",
    parse_dates=["date"]
)

print("=" * 60)
print("OVERALL SALES TREND ANALYSIS")
print("=" * 60)

# 3. Calculate total sales by date
daily_sales = (
    df.groupby("date")["sales"]
    .sum()
    .reset_index()
)

# 4. Calculate monthly sales
monthly_sales = (
    daily_sales
    .set_index("date")["sales"]
    .resample("MS")
    .sum()
    .reset_index()
)

# 5. Calculate average daily sales
average_daily_sales = daily_sales["sales"].mean()

print("\nSALES SUMMARY")
print("Total sales:", round(df["sales"].sum(), 2))
print("Average daily sales:", round(average_daily_sales, 2))
print("Highest daily sales:",
      round(daily_sales["sales"].max(), 2))
print("Lowest daily sales:",
      round(daily_sales["sales"].min(), 2))

# 6. Identify the highest-sales date
highest_sales_day = daily_sales.loc[
    daily_sales["sales"].idxmax()
]

print("\nHIGHEST SALES DAY")
print("Date:", highest_sales_day["date"].date())
print("Sales:", round(highest_sales_day["sales"], 2))

# 7. Display monthly sales
print("\nFIRST 5 MONTHLY SALES RECORDS")
print(monthly_sales.head())

# 8. Create monthly sales chart
plt.figure(figsize=(14, 6))

plt.plot(
    monthly_sales["date"],
    monthly_sales["sales"],
    marker="o"
)

plt.title("Monthly Retail Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# 9. Save chart
chart_path = figure_folder / "monthly_sales_trend.png"

plt.savefig(chart_path)
plt.show()

print("\nChart saved at:", chart_path)

# 10. Save monthly sales summary
monthly_sales.to_csv(
    data_folder / "monthly_sales_summary.csv",
    index=False
)

print("\nMonthly sales summary saved successfully.")

print("\n" + "=" * 60)
print("SALES TREND ANALYSIS COMPLETED")
print("=" * 60)