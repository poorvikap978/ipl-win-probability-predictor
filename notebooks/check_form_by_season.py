
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parent.parent
FILE = PROJECT_DIR / "data" / "processed" / "match_features.csv"

df = pd.read_csv(FILE, parse_dates=["match_date"])

df["season"] = df["match_date"].dt.year

df["form_difference"] = (
    df["team1_recent_win_pct"]
    - df["team2_recent_win_pct"]
)

test = df[df["match_date"] >= "2024-01-01"].copy()

print("Recent-form performance by season")

for season, group in test.groupby("season"):
    y = group["team1_won"]
    score = group["form_difference"]

    auc = roc_auc_score(y, score)

    print(f"\nSeason: {season}")
    print("Matches:", len(group))
    print("Actual Team 1 win rate:", round(y.mean(), 4))
    print("Form difference AUC:", round(auc, 4))
