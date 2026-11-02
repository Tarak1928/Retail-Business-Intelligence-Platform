 
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("data/processed/train_model_ready.csv")

# --------------------------------------------------
# 2. CLASSIFY PROMOTED AND NON-PROMOTED RECORDS
# --------------------------------------------------

df["promotion_status"] = df["onpromotion"].apply(
    lambda x: "Promoted" if x > 0 else "Not Promoted"
)

# --------------------------------------------------
# 3. ANALYZE PROMOTION PERFORMANCE
# --------------------------------------------------

promotion_analysis = (
    df.groupby("promotion_status")
    .agg(
        total_sales=("sales", "sum"),
        average_sales=("sales", "mean"),
        total_records=("sales", "count"),
        average_promotion_count=("onpromotion", "mean")
    )
    .reset_index()
)

print("\n" + "=" * 60)
print("PROMOTION IMPACT ANALYSIS")
print("=" * 60)

print("\nPROMOTION PERFORMANCE")

print(promotion_analysis.to_string(index=False))

# --------------------------------------------------
# 4. CALCULATE SALES DIFFERENCE
# --------------------------------------------------

promoted_sales = promotion_analysis.loc[
    promotion_analysis["promotion_status"] == "Promoted",
    "average_sales"
].iloc[0]

non_promoted_sales = promotion_analysis.loc[
    promotion_analysis["promotion_status"] == "Not Promoted",
    "average_sales"
].iloc[0]

difference = promoted_sales - non_promoted_sales

print("\nAVERAGE SALES COMPARISON")

print("Promoted average sales:", promoted_sales)
print("Non-promoted average sales:", non_promoted_sales)
print("Average sales difference:", difference)

# --------------------------------------------------
# 5. ANALYZE PROMOTIONS BY PRODUCT FAMILY
# --------------------------------------------------

family_promotion = (
    df.groupby("family")
    .agg(
        total_promotions=("onpromotion", "sum"),
        average_sales=("sales", "mean")
    )
    .sort_values(
        by="total_promotions",
        ascending=False
    )
)

print("\nTOP 10 PRODUCT FAMILIES BY PROMOTION COUNT")

print(family_promotion.head(10).to_string())

# --------------------------------------------------
# 6. CREATE PROMOTION COMPARISON CHART
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    promotion_analysis["promotion_status"],
    promotion_analysis["average_sales"]
)

plt.title("Average Sales: Promoted vs Not Promoted")
plt.xlabel("Promotion Status")
plt.ylabel("Average Sales")

plt.tight_layout()

plt.savefig(
    "reports/figures/promotion_sales_comparison.png",
    dpi=300
)

plt.show()

# --------------------------------------------------
# 7. SAVE REPORTS
# --------------------------------------------------

promotion_analysis.to_csv(
    "data/processed/promotion_analysis.csv",
    index=False
)

family_promotion.to_csv(
    "data/processed/family_promotion_analysis.csv"
)

print("\nPromotion comparison chart saved.")
print("Promotion analysis reports saved.")

print("\nPROMOTION ANALYSIS COMPLETED")
