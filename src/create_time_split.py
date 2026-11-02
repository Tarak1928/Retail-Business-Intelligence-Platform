
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data/processed/train_encoded.csv"

TRAIN_OUTPUT = PROJECT_ROOT / "data/processed/train_period.csv"
VALIDATION_OUTPUT = PROJECT_ROOT / "data/processed/validation_period.csv"

TRAIN_END = pd.Timestamp("2017-07-30")
VALIDATION_START = pd.Timestamp("2017-07-31")
VALIDATION_END = pd.Timestamp("2017-08-15")

CHUNK_SIZE = 100_000

# Lag features must be regenerated recursively
# during validation, not taken from actual sales.
LAG_COLUMNS = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_30",
    "sales_rolling_mean_7",
]

# --------------------------------------------------
# 2. Prepare output files
# --------------------------------------------------

for path in [TRAIN_OUTPUT, VALIDATION_OUTPUT]:
    if path.exists():
        path.unlink()

train_rows = 0
validation_rows = 0

print("Starting chronological data splitting...")

# --------------------------------------------------
# 3. Process the dataset in chunks
# --------------------------------------------------

for chunk in pd.read_csv(INPUT_PATH, chunksize=CHUNK_SIZE):

    chunk["date"] = pd.to_datetime(chunk["date"])

    # Historical training period
    train_chunk = chunk[chunk["date"] <= TRAIN_END]

    if not train_chunk.empty:
        train_chunk.to_csv(
            TRAIN_OUTPUT,
            mode="a",
            header=not TRAIN_OUTPUT.exists(),
            index=False
        )

        train_rows += len(train_chunk)

    # Validation period
    validation_chunk = chunk[
        (chunk["date"] >= VALIDATION_START)
        & (chunk["date"] <= VALIDATION_END)
    ].copy()

    if not validation_chunk.empty:

        # Remove precomputed lags because they contain
        # actual validation-period sales information.
        validation_chunk = validation_chunk.drop(
            columns=LAG_COLUMNS
        )

        validation_chunk.to_csv(
            VALIDATION_OUTPUT,
            mode="a",
            header=not VALIDATION_OUTPUT.exists(),
            index=False
        )

        validation_rows += len(validation_chunk)

# --------------------------------------------------
# 4. Display results
# --------------------------------------------------

print("\nChronological splitting completed!")

print("Training rows:", train_rows)
print("Validation rows:", validation_rows)

print("\nTraining file:", TRAIN_OUTPUT)
print("Validation file:", VALIDATION_OUTPUT)

print("\nExpected training rows: 2,972,376")
print("Expected validation rows: 28,512")

assert train_rows == 2_972_376
assert validation_rows == 28_512

print("\nAll split validation checks passed!")
