
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

clean = pd.read_csv(DATA_DIR / "clean_matches.csv")
features = pd.read_csv(DATA_DIR / "match_features.csv")

# Check row counts and match IDs.
print("Clean matches:", len(clean))
print("Feature rows:", len(features))
print("Duplicate feature match IDs:", features["match_id"].duplicated().sum())

clean_ids = set(clean["match_id"])
feature_ids = set(features["match_id"])

print("IDs missing from features:", len(clean_ids - feature_ids))
print("Extra IDs in features:", len(feature_ids - clean_ids))

# Compare target labels by match ID.
check = clean[["match_id", "team1_won"]].merge(
    features[["match_id", "team1_won"]],
    on="match_id",
    how="outer",
    suffixes=("_clean", "_features"),
    indicator=True,
)

matched = check[check["_merge"] == "both"]
mismatches = (
    matched["team1_won_clean"] != matched["team1_won_features"]
)

print("Matched IDs:", len(matched))
print("Target mismatches by match ID:", int(mismatches.sum()))
print("Unmatched IDs:", int((check["_merge"] != "both").sum()))

# Check the date range of the feature file.
features["match_date"] = pd.to_datetime(features["match_date"])
print("Feature date range:",
      features["match_date"].min(),
      "to",
      features["match_date"].max())
