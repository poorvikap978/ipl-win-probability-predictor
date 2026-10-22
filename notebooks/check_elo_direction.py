
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "processed" / "match_features.csv"

df = pd.read_csv(DATA_FILE, parse_dates=["match_date"])
test = df[df["match_date"] >= "2024-01-01"].copy()

test["elo_favorite_won"] = (
    (test["elo_difference"] > 0) & (test["team1_won"] == 1)
) | (
    (test["elo_difference"] < 0) & (test["team1_won"] == 0)
)

test["elo_favorite_exists"] = test["elo_difference"] != 0

non_ties = test[test["elo_favorite_exists"]]

print("Test matches:", len(test))
print("Matches with unequal Elo ratings:", len(non_ties))
print(
    "Higher-rated team win rate:",
    round(non_ties["elo_favorite_won"].mean(), 4),
)
print(
    "Mean Elo difference when Team 1 wins:",
    round(test.loc[test["team1_won"] == 1, "elo_difference"].mean(), 3),
)
print(
    "Mean Elo difference when Team 1 loses:",
    round(test.loc[test["team1_won"] == 0, "elo_difference"].mean(), 3),
)
