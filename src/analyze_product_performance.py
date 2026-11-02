import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Locate folders
project_root = Path(__file__).resolve().parents[1]

data_folder = project_root / "data" / "processed"
figure_folder = project_root / "reports" / "figures"

figure_folder.mkdir(parents=True, exist_ok=True)

# Load training data
df = pd.read_csv(
    data_folder / "train_model_ready.csv"
)

print("=" * 60)
print("PRODUCT FAMILY PERFORMANCE ANALYSIS")
print("=" * 60)

# Calculate sales performance by product family
product_performance = (
    df.groupby("family")
    .agg(
        total_sales=("sales", "sum"),
        average_sales=("sales", "mean"),
        sales_records=("sales", "count")
    )
    .reset_index()
)

# Sort by total sales
product_performance = product_performance.sort_values(
    by="total_sales",
    ascending=False
)

# Top 10 product families
print("\nTOP 10 PRODUCT FAMILIES")
print(product_performance.head(10))

# Bottom 10 product families
print("\nBOTTOM 10 PRODUCT FAMILIES")
print(product_performance.tail(10))

# Identify the highest and lowest performers
top_product = product_performance.iloc[0]
bottom_product = product_performance.iloc[-1]

print("\nBEST PERFORMING PRODUCT FAMILY")
print("Family:", top_product["family"])
print("Total sales:", round(top_product["total_sales"], 2))

print("\nLOWEST PERFORMING PRODUCT FAMILY")
print("Family:", bottom_product["family"])
print("Total sales:", round(bottom_product["total_sales"], 2))

# Create top 10 bar chart
top_10 = product_performance.head(10)

plt.figure(figsize=(12, 6))

plt.bar(
    top_10["family"],
    top_10["total_sales"]
)

plt.title("Top 10 Product Families by Total Sales")
plt.xlabel("Product Family")
plt.ylabel("Total Sales")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Save chart
chart_path = figure_folder / "top_product_families.png"

plt.savefig(chart_path)
plt.show()

# Save complete product performance report
product_performance.to_csv(
    data_folder / "product_performance.csv",
    index=False
)

print("\nChart saved:", chart_path)
print("Product report saved successfully.")

print("\n" + "=" * 60)
print("PRODUCT ANALYSIS COMPLETED")
print("=" * 60)