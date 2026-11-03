
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "lightgbm_sales_model.txt"

DATA_PATH = BASE_DIR / "data" / "processed" / "train_period.csv"

REPORT_DIR = BASE_DIR / "reports"

FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load trained LightGBM model
# --------------------------------------------------

print("Loading LightGBM model...")

model = lgb.Booster(
    model_file=str(MODEL_PATH)
)

print("Model loaded successfully!")


# --------------------------------------------------
# 3. Recover original feature names
# --------------------------------------------------

print("\nReading training dataset columns...")

sample = pd.read_csv(DATA_PATH, nrows=1)

excluded_columns = [
    "id",
    "date",
    "sales",
    "dcoilwtico"
]

original_feature_names = [
    column
    for column in sample.columns
    if column not in excluded_columns
]

if len(original_feature_names) != model.num_feature():
    raise ValueError(
        "Feature count does not match the trained model. "
        "Check the feature selection in train_lightgbm.py."
    )

print("Total features:", len(original_feature_names))


# --------------------------------------------------
# 4. Extract feature importance
# --------------------------------------------------

print("\nCalculating feature importance...")

importance = model.feature_importance(
    importance_type="gain"
)

importance_df = pd.DataFrame({
    "Feature": original_feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df["Importance_Percentage"] = (
    importance_df["Importance"]
    / importance_df["Importance"].sum()
) * 100

print("\nTOP 20 IMPORTANT FEATURES")
print("-------------------------")

print(
    importance_df.head(20).to_string(index=False)
)


# --------------------------------------------------
# 5. Save feature importance report
# --------------------------------------------------

output_csv = REPORT_DIR / "feature_importance.csv"

importance_df.to_csv(
    output_csv,
    index=False
)

print("\nFeature importance saved to:")
print(output_csv)


# --------------------------------------------------
# 6. Visualize top 20 features
# --------------------------------------------------

top_features = importance_df.head(20).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(12, 8))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance (Gain)")
plt.ylabel("Features")

plt.title(
    "Top 20 Feature Importance - LightGBM Model"
)

plt.tight_layout()

chart_path = FIGURE_DIR / "feature_importance.png"

plt.savefig(
    chart_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFeature importance chart saved to:")
print(chart_path)

print("\nFeature importance analysis completed!")
