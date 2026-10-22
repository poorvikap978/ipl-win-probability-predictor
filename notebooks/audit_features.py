
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

matches = pd.read_csv(DATA_DIR / "clean_matches.csv")
features = pd.read_csv(DATA_DIR / "match_features.csv")

print("CLEANED MATCHES")
print("Columns:", matches.columns.tolist())
print(matches.head(3).to_string(index=False))

print("\nFEATURE DATA")
print("Columns:", features.columns.tolist())

print("\nTARGET DISTRIBUTION")
print(features["team1_won"].value_counts(normalize=True).sort_index())

print("\nELO FEATURE SUMMARY")
elo_cols = ["team1_elo", "team2_elo", "elo_difference"]
print(features[elo_cols].describe().round(3))

print("\nELO DIFFERENCE VS TARGET")
print(
    features.groupby("team1_won")["elo_difference"]
    .agg(["count", "mean", "median"])
    .round(3)
)
