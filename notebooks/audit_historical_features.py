
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

clean = pd.read_csv(DATA_DIR / "clean_matches.csv", parse_dates=["match_date"])
features = pd.read_csv(DATA_DIR / "match_features.csv", parse_dates=["match_date"])

clean = clean.sort_values(["match_date", "match_id"]).reset_index(drop=True)
features = features.sort_values(["match_date", "match_id"]).reset_index(drop=True)

print("=== BASIC CHECKS ===")
print("Rows:", len(features))
print("Missing values in feature columns:")
print(features.isna().sum().to_string())

print("\n=== RECENT FORM RANGE ===")
for col in ["team1_recent_win_pct", "team2_recent_win_pct"]:
    print(
        col,
        "min =", features[col].min(),
        "max =", features[col].max(),
        "out-of-range =", int((~features[col].between(0, 1)).sum()),
    )

print("\n=== OVERALL WIN RATE RANGE ===")
for col in ["team1_overall_win_pct", "team2_overall_win_pct"]:
    print(
        col,
        "min =", features[col].min(),
        "max =", features[col].max(),
        "out-of-range =", int((~features[col].between(0, 1)).sum()),
    )

print("\n=== HEAD-TO-HEAD AND VENUE COUNTS ===")
count_cols = [
    "head_to_head_matches",
    "team1_venue_matches",
    "team2_venue_matches",
    "team1_recent_matches",
    "team2_recent_matches",
]
for col in count_cols:
    print(
        col,
        "min =", features[col].min(),
        "max =", features[col].max(),
        "negative counts =", int((features[col] < 0).sum()),
    )

print("\n=== SAME-DATE MATCHES ===")
matches_per_date = clean.groupby("match_date").size()
multi_match_dates = matches_per_date[matches_per_date > 1]

print("Dates with multiple matches:", len(multi_match_dates))
print("Matches on those dates:", int(multi_match_dates.sum()))

print("\n=== HISTORICAL FRANCHISES ===")
print(
    sorted(
        set(features["team1_name"].dropna())
        | set(features["team2_name"].dropna())
    )
)
