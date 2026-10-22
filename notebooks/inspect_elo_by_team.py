
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "processed" / "match_features.csv"

df = pd.read_csv(DATA_FILE, parse_dates=["match_date"])
test = df[df["match_date"] >= "2024-01-01"].copy()

team1 = test[[
    "season", "team1_name", "team1_elo", "team1_won"
]].rename(columns={
    "team1_name": "team",
    "team1_elo": "elo",
    "team1_won": "won",
})

team2 = test[[
    "season", "team2_name", "team2_elo", "team1_won"
]].rename(columns={
    "team2_name": "team",
    "team2_elo": "elo",
    "team1_won": "won",
})

team2["won"] = 1 - team2["won"]

appearances = pd.concat([team1, team2], ignore_index=True)

summary = appearances.groupby(["season", "team"]).agg(
    matches=("won", "size"),
    wins=("won", "sum"),
    mean_elo=("elo", "mean"),
    win_rate=("won", "mean"),
).reset_index()

summary["win_rate"] = summary["win_rate"].round(3)
summary["mean_elo"] = summary["mean_elo"].round(1)

print(summary.sort_values(
    ["season", "mean_elo"],
    ascending=[True, False]
).to_string(index=False))

from scipy.stats import pointbiserialr

correlation, p_value = pointbiserialr(
    test["team1_won"],
    test["elo_difference"],
)

print("\nElo difference vs. Team 1 win")
print("Correlation:", round(correlation, 4))
print("P-value:", round(p_value, 4))