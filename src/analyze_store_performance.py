
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. LOAD THE DATASET
# --------------------------------------------------

df = pd.read_csv("data/processed/train_model_ready.csv")

# --------------------------------------------------
# 2. CONVERT DATE COLUMN
# --------------------------------------------------

df["date"] = pd.to_datetime(df["date"])

# --------------------------------------------------
# 3. CALCULATE STORE PERFORMANCE
# --------------------------------------------------

store_performance = (
    df.groupby("store_nbr")
    .agg(
        total_sales=("sales", "sum"),
        average_sales=("sales", "mean"),
        sales_records=("sales", "count")
    )
    .reset_index()
)

# Sort stores by total sales
store_performance = store_performance.sort_values(
    by="total_sales",
    ascending=False
)

# --------------------------------------------------
# 4. DISPLAY TOP 10 STORES
# --------------------------------------------------

print("\n" + "=" * 60)
print("STORE PERFORMANCE ANALYSIS")
print("=" * 60)

print("\nTOP 10 STORES")

print(store_performance.head(10).to_string(index=False))

# --------------------------------------------------
# 5. DISPLAY BOTTOM 10 STORES
# --------------------------------------------------

print("\nBOTTOM 10 STORES")

print(
    store_performance.tail(10)
    .sort_values("total_sales")
    .to_string(index=False)
)

# --------------------------------------------------
# 6. IDENTIFY BEST AND LOWEST STORES
# --------------------------------------------------

best_store = store_performance.iloc[0]
lowest_store = store_performance.iloc[-1]

print("\nBEST PERFORMING STORE")
print("Store:", best_store["store_nbr"])
print("Total Sales:", best_store["total_sales"])

print("\nLOWEST PERFORMING STORE")
print("Store:", lowest_store["store_nbr"])
print("Total Sales:", lowest_store["total_sales"])

# --------------------------------------------------
# 7. CREATE TOP 10 STORE CHART
# --------------------------------------------------

top_10 = store_performance.head(10)

plt.figure(figsize=(12, 6))

plt.bar(
    top_10["store_nbr"].astype(str),
    top_10["total_sales"]
)

plt.title("Top 10 Stores by Total Sales")
plt.xlabel("Store Number")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "reports/figures/top_10_stores.png",
    dpi=300
)

plt.show()

# --------------------------------------------------
# 8. SAVE STORE PERFORMANCE REPORT
# --------------------------------------------------

store_performance.to_csv(
    "data/processed/store_performance.csv",
    index=False
)

print("\nChart saved successfully.")
print("Store performance report saved successfully.")

print("\nSTORE ANALYSIS COMPLETED")


import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("data/processed/train_model_ready.csv")

df["date"] = pd.to_datetime(df["date"])

# --------------------------------------------------
# 2. CALCULATE TOTAL DAILY SALES
# --------------------------------------------------

daily_sales = (
    df.groupby("date")["sales"]
    .sum()
    .reset_index()
)

daily_sales.columns = ["date", "total_sales"]

daily_sales["day_name"] = daily_sales["date"].dt.day_name()
daily_sales["day_number"] = daily_sales["date"].dt.dayofweek
daily_sales["month"] = daily_sales["date"].dt.month
daily_sales["month_name"] = daily_sales["date"].dt.month_name()
daily_sales["year"] = daily_sales["date"].dt.year

daily_sales["day_type"] = daily_sales["day_number"].apply(
    lambda day: "Weekend" if day >= 5 else "Weekday"
)

# --------------------------------------------------
# 3. DAILY SALES ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 60)
print("TIME-BASED SALES ANALYSIS")
print("=" * 60)

print("\nDAILY SALES SUMMARY")

print("Total daily records:", len(daily_sales))
print("Average daily sales:", daily_sales["total_sales"].mean())
print("Highest daily sales:", daily_sales["total_sales"].max())
print("Lowest daily sales:", daily_sales["total_sales"].min())

# --------------------------------------------------
# 4. MONTHLY SALES ANALYSIS
# --------------------------------------------------

monthly_sales = (
    daily_sales.set_index("date")["total_sales"]
    .resample("MS")
    .sum()
    .reset_index()
)

monthly_sales.columns = ["month", "total_sales"]

print("\nMONTHLY SALES SUMMARY")

print(monthly_sales.head(12).to_string(index=False))

# --------------------------------------------------
# 5. WEEKDAY SALES ANALYSIS
# --------------------------------------------------

weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekday_sales = (
    daily_sales.groupby("day_name")
    .agg(
        total_sales=("total_sales", "sum"),
        average_daily_sales=("total_sales", "mean"),
        number_of_days=("date", "count")
    )
    .reindex(weekday_order)
    .reset_index()
)

print("\nWEEKDAY PERFORMANCE")

print(weekday_sales.to_string(index=False))

best_day = weekday_sales.loc[
    weekday_sales["average_daily_sales"].idxmax()
]

lowest_day = weekday_sales.loc[
    weekday_sales["average_daily_sales"].idxmin()
]

print("\nBEST PERFORMING WEEKDAY")
print("Day:", best_day["day_name"])
print("Average sales:", best_day["average_daily_sales"])

print("\nLOWEST PERFORMING WEEKDAY")
print("Day:", lowest_day["day_name"])
print("Average sales:", lowest_day["average_daily_sales"])

# --------------------------------------------------
# 6. WEEKDAY VS WEEKEND ANALYSIS
# --------------------------------------------------

day_type_sales = (
    daily_sales.groupby("day_type")
    .agg(
        total_sales=("total_sales", "sum"),
        average_daily_sales=("total_sales", "mean"),
        number_of_days=("date", "count")
    )
    .reset_index()
)

print("\nWEEKDAY VS WEEKEND SALES")

print(day_type_sales.to_string(index=False))

# --------------------------------------------------
# 7. MONTH-OF-YEAR SALES ANALYSIS
# --------------------------------------------------

month_performance = (
    daily_sales.groupby("month")
    .agg(
        total_sales=("total_sales", "sum"),
        average_daily_sales=("total_sales", "mean")
    )
    .reset_index()
)

month_performance["month_name"] = (
    month_performance["month"].apply(
        lambda month: pd.Timestamp(
            year=2020, month=month, day=1
        ).month_name()
    )
)

print("\nMONTH-OF-YEAR PERFORMANCE")

print(month_performance.to_string(index=False))

# --------------------------------------------------
# 8. CREATE MONTHLY SALES CHART
# --------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    monthly_sales["month"],
    monthly_sales["total_sales"],
    marker="o"
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "reports/figures/monthly_sales_time_analysis.png",
    dpi=300
)

plt.show()

# --------------------------------------------------
# 9. CREATE WEEKDAY SALES CHART
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    weekday_sales["day_name"],
    weekday_sales["average_daily_sales"]
)

plt.title("Average Daily Sales by Weekday")
plt.xlabel("Day of the Week")
plt.ylabel("Average Daily Sales")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "reports/figures/weekday_sales_analysis.png",
    dpi=300
)

plt.show()

# --------------------------------------------------
# 10. CREATE WEEKDAY VS WEEKEND CHART
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    day_type_sales["day_type"],
    day_type_sales["average_daily_sales"]
)

plt.title("Average Daily Sales: Weekday vs Weekend")
plt.xlabel("Day Type")
plt.ylabel("Average Daily Sales")

plt.tight_layout()

plt.savefig(
    "reports/figures/weekday_vs_weekend_sales.png",
    dpi=300
)

plt.show()

# --------------------------------------------------
# 11. SAVE ANALYSIS REPORTS
# --------------------------------------------------

daily_sales.to_csv(
    "data/processed/daily_sales_summary.csv",
    index=False
)

monthly_sales.to_csv(
    "data/processed/monthly_sales_time_analysis.csv",
    index=False
)

weekday_sales.to_csv(
    "data/processed/weekday_sales_analysis.csv",
    index=False
)

day_type_sales.to_csv(
    "data/processed/weekday_weekend_analysis.csv",
    index=False
)

month_performance.to_csv(
    "data/processed/month_performance.csv",
    index=False
)

print("\nAll charts saved successfully.")
print("All time-based sales reports saved successfully.")

print("\nTIME-BASED SALES ANALYSIS COMPLETED")
