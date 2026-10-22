
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "processed" / "match_features.csv"

df = pd.read_csv(DATA_FILE, parse_dates=["match_date"])
df = df[df["match_date"] >= "2024-01-01"].copy()

# Elo-only probability that Team 1 wins.
df["elo_probability"] = 1 / (
    1 + 10 ** (-df["elo_difference"] / 400)
)

df["elo_prediction"] = (
    df["elo_probability"] >= 0.5
).astype(int)

for season, group in df.groupby("season", sort=True):
    y = group["team1_won"]
    p = group["elo_probability"]
    pred = group["elo_prediction"]

    print(f"\n===== Season {season} =====")
    print("Matches:", len(group))
    print("Team 1 win rate:", round(y.mean(), 4))
    print("Elo accuracy:", round(accuracy_score(y, pred), 4))
    print("Elo log loss:", round(log_loss(y, p, labels=[0, 1]), 4))
    print("Elo Brier score:", round(brier_score_loss(y, p), 4))

    if y.nunique() == 2:
        print("Elo ROC-AUC:", round(roc_auc_score(y, p), 4))
    else:
        print("Elo ROC-AUC: unavailable (only one target class)")
