
import pandas as pd
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


matches_file = DATA_DIR / "ipl_matches_data.csv"
deliveries_file = DATA_DIR / "ball_by_ball_data.csv"


if not matches_file.exists():
    raise FileNotFoundError(f"Matches file not found: {matches_file}")

if not deliveries_file.exists():
    raise FileNotFoundError(f"Deliveries file not found: {deliveries_file}")


matches = pd.read_csv(matches_file)
deliveries = pd.read_csv(deliveries_file)


print("\n========== MATCHES DATASET ==========")
print("Shape:", matches.shape)

print("\nColumn names:")
print(matches.columns.tolist())

print("\nFirst 5 rows:")
print(matches.head().to_string())

print("\nData types:")
print(matches.dtypes)

print("\nMissing values:")
print(matches.isnull().sum())

print("\nDuplicate rows:", matches.duplicated().sum())


print("\n========== DELIVERIES DATASET ==========")
print("Shape:", deliveries.shape)

print("\nColumn names:")
print(deliveries.columns.tolist())

print("\nFirst 5 rows:")
print(deliveries.head().to_string())

print("\nData types:")
print(deliveries.dtypes)

print("\nMissing values:")
print(deliveries.isnull().sum())

print("\nDuplicate rows:", deliveries.duplicated().sum())

print("\n========== INSPECTION COMPLETE ==========")
