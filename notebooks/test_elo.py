
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

cutoff = pd.Timestamp("2024-01-01")
test = df[df["match_date"] >= cutoff].copy()

y_test = test["team1_won"]

# Elo-based expected probability that Team 1 wins.
elo_difference = test["elo_difference"]
probabilities = 1 / (1 + 10 ** (-elo_difference / 400))

predictions = (probabilities >= 0.5).astype(int)

print("Test matches:", len(test))
print("Actual Team 1 win rate:", round(y_test.mean(), 4))
print("Mean Elo win probability:", round(probabilities.mean(), 4))
print("Elo-only accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Elo-only ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
print("Elo-only log loss:", round(log_loss(y_test, probabilities), 4))
print("Elo-only Brier score:", round(brier_score_loss(y_test, probabilities), 4))
