
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
FILE = PROJECT_DIR / "data" / "processed" / "match_features.csv"

df = pd.read_csv(FILE, parse_dates=["match_date"])
df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

cutoff = pd.Timestamp("2024-01-01")
train = df[df["match_date"] < cutoff].copy()
test = df[df["match_date"] >= cutoff].copy()

print("Dataset shape:", df.shape)
print("Training matches:", len(train))
print("Testing matches:", len(test))

print("\nTarget distribution in test set:")
print(test["team1_won"].value_counts().sort_index())

print("\nFeature summary:")
numeric = [
    "team1_won_toss",
    "team1_recent_win_pct",
    "team2_recent_win_pct",
    "team1_recent_matches",
    "team2_recent_matches",
    "team1_overall_win_pct",
    "team2_overall_win_pct",
    "team1_h2h_win_pct",
    "head_to_head_matches",
    "team1_venue_win_pct",
    "team1_venue_matches",
    "team2_venue_win_pct",
    "team2_venue_matches",
]

print(df[numeric].describe().round(3))

print("\nFeature correlations with target:")
print(
    df[numeric + ["team1_won"]]
    .corr(numeric_only=True)["team1_won"]
    .sort_values(ascending=False)
    .round(3)
)

print("\nTest accuracy from simple baselines:")
majority = test["team1_won"].value_counts(normalize=True).max()
team1_toss_baseline = (
    (test["team1_won_toss"] == test["team1_won"]).mean()
)

print("Always predict test majority:", round(majority, 4))
print("Predict Team 1 wins if it won toss:", round(team1_toss_baseline, 4))
