
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
FILE = PROJECT_DIR / "data" / "processed" / "match_features.csv"

df = pd.read_csv(FILE, parse_dates=["match_date"])

cutoff = pd.Timestamp("2024-01-01")
test = df[df["match_date"] >= cutoff].copy()

actual = test["team1_won"].astype(int)

# Elo probability: expected result for Team 1.
# The scale factor 400 is standard for the conventional Elo formula.
elo_probability = 1 / (
    1 + 10 ** (-test["elo_difference"] / 400)
)

# Constant 50% baseline.
baseline_probability = pd.Series(0.5, index=test.index)

for name, probability in [
    ("50% baseline", baseline_probability),
    ("Elo probability", elo_probability),
]:
    prediction = (probability >= 0.5).astype(int)

    print(f"\n=== {name} ===")
    print("Accuracy:", round(accuracy_score(actual, prediction), 4))
    print("ROC-AUC:", round(roc_auc_score(actual, probability), 4))
    print("Log loss:", round(log_loss(actual, probability), 4))
    print("Brier score:", round(brier_score_loss(actual, probability), 4))

print("\nMean Elo probability:", round(elo_probability.mean(), 4))
print("Actual Team 1 win rate:", round(actual.mean(), 4))
