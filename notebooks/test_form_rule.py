
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


recent_diff = (
    test["team1_recent_win_pct"] -
    test["team2_recent_win_pct"]
)


overall_diff = (
    test["team1_overall_win_pct"] -
    test["team2_overall_win_pct"]
)

score = recent_diff.copy()
ties = score == 0
score.loc[ties] = overall_diff.loc[ties]


probability = pd.Series(0.5, index=test.index)
probability.loc[score > 0] = 0.60
probability.loc[score < 0] = 0.40

prediction = (probability >= 0.5).astype(int)

print("Test matches:", len(test))
print("Actual Team 1 win rate:", round(actual.mean(), 4))
print("Rule accuracy:", round(accuracy_score(actual, prediction), 4))
print("ROC-AUC:", round(roc_auc_score(actual, probability), 4))
print("Log loss:", round(log_loss(actual, probability), 4))
print("Brier score:", round(brier_score_loss(actual, probability), 4))
print("Exact ties:", int((score == 0).sum()))
