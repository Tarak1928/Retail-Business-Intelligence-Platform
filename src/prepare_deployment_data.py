
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEPLOYMENT_DIR = PROJECT_ROOT / "data" / "deployment"

DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

# Prepare test metadata
print("Preparing test metadata...")

test = pd.read_csv(
    PROCESSED_DIR / "test_model_ready.csv",
    usecols=["id", "date", "store_nbr", "family"]
)

test.to_csv(
    DEPLOYMENT_DIR / "test_metadata.csv",
    index=False
)

print(f"Test records: {len(test):,}")

# Prepare historical daily sales
print("Preparing historical daily sales...")

historical = pd.read_csv(
    PROCESSED_DIR / "train_model_ready.csv",
    usecols=["date", "sales"]
)

historical["date"] = pd.to_datetime(historical["date"])

daily_sales = (
    historical.groupby("date", as_index=False)["sales"]
    .sum()
)

daily_sales.to_csv(
    DEPLOYMENT_DIR / "historical_daily_sales.csv",
    index=False,
    date_format="%Y-%m-%d"
)

print(f"Historical daily records: {len(daily_sales):,}")

# Display generated files
print("\nDeployment datasets created successfully.")

for file_path in DEPLOYMENT_DIR.iterdir():
    if file_path.is_file():
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"{file_path.name}: {size_mb:.2f} MB")
