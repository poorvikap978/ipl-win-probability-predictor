
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
FEATURE_FILE = PROJECT_DIR / "data" / "processed" / "match_features.csv"
CLEAN_FILE = PROJECT_DIR / "data" / "processed" / "clean_matches.csv"

features = pd.read_csv(FEATURE_FILE, parse_dates=["match_date"])
clean = pd.read_csv(CLEAN_FILE, parse_dates=["match_date"])

print("=== FEATURE FILE CHECK ===")
print("Feature rows:", len(features))
print("Unique match IDs:", features["match_id"].nunique())
print("Duplicate match IDs:", features["match_id"].duplicated().sum())

print("\n=== TARGET CHECK ===")
print("Target distribution:")
print(features["team1_won"].value_counts().sort_index())

print("\n=== FIRST MATCHES ===")
print(
    features[
        [
            "match_date",
            "match_id",
            "team1_name",
            "team2_name",
            "team1_recent_matches",
            "team2_recent_matches",
            "team1_elo",
            "team2_elo",
            "team1_won",
        ]
    ].head(10).to_string(index=False)
)

print("\n=== ALIGNMENT CHECK ===")
merged = clean[["match_id", "team1_won"]].merge(
    features[["match_id", "team1_won"]],
    on="match_id",
    how="outer",
    suffixes=("_clean", "_features"),
    indicator=True,
)

print("Unmatched IDs:", (merged["_merge"] != "both").sum())

matched = merged[merged["_merge"] == "both"]
print(
    "Target mismatches:",
    (matched["team1_won_clean"] != matched["team1_won_features"]).sum(),
)
