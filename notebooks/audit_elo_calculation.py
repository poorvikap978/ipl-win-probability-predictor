
import pandas as pd
from pathlib import Path
from scipy.stats import pointbiserialr

PROJECT_DIR = Path(__file__).resolve().parent.parent
FILE = PROJECT_DIR / "data" / "processed" / "match_features.csv"

df = pd.read_csv(FILE, parse_dates=["match_date"])
df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

cutoff = pd.Timestamp("2024-01-01")
test = df[df["match_date"] >= cutoff].copy()

# 1. Check whether Elo difference is calculated correctly.
df["calculated_elo_difference"] = df["team1_elo"] - df["team2_elo"]

difference = (
    df["elo_difference"] - df["calculated_elo_difference"]
).abs()

print("=== Elo calculation check ===")
print("Largest difference:", round(difference.max(), 8))
print("Rows with inconsistent Elo difference:", int((difference > 0.001).sum()))

# 2. Check whether higher Elo is associated with Team 1 winning.
correlation, p_value = pointbiserialr(
    test["team1_won"],
    test["elo_difference"]
)

print("\n=== Test-set Elo direction ===")
print("Matches:", len(test))
print("Correlation:", round(correlation, 4))
print("P-value:", round(p_value, 4))
print(
    "Team 1 win rate when its Elo is higher:",
    round(test.loc[test["elo_difference"] > 0, "team1_won"].mean(), 4)
)
print(
    "Team 1 win rate when its Elo is lower:",
    round(test.loc[test["elo_difference"] < 0, "team1_won"].mean(), 4)
)

# 3. Examine the direction season by season.
print("\n=== Elo direction by season ===")

for season, group in test.groupby("season"):
    if group["elo_difference"].nunique() < 2:
        continue

    corr = group["team1_won"].corr(group["elo_difference"])
    higher = group[group["elo_difference"] > 0]
    lower = group[group["elo_difference"] < 0]

    print(
        season,
        "| matches:", len(group),
        "| correlation:", round(corr, 3),
        "| Team 1 win rate when Elo higher:",
        round(higher["team1_won"].mean(), 3) if len(higher) else "N/A",
        "| when Elo lower:",
        round(lower["team1_won"].mean(), 3) if len(lower) else "N/A",
    )
