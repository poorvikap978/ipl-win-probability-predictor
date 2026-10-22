
import pandas as pd
from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
FILE = PROJECT_DIR / "data" / "processed" / "match_features.csv"

df = pd.read_csv(FILE, parse_dates=["match_date"])
df = df.sort_values(["match_date", "match_id"])

train = df[df["match_date"] < "2024-01-01"]
test = df[df["match_date"] >= "2024-01-01"]

features = ["elo_difference"]

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000),
)

model.fit(train[features], train["team1_won"])

probabilities = model.predict_proba(test[features])[:, 1]
predictions = (probabilities >= 0.5).astype(int)

print("Training matches:", len(train))
print("Test matches:", len(test))
print("Accuracy:", round(
    accuracy_score(test["team1_won"], predictions), 4
))
print("ROC-AUC:", round(
    roc_auc_score(test["team1_won"], probabilities), 4
))
print("Log loss:", round(
    log_loss(test["team1_won"], probabilities), 4
))
print("Brier score:", round(
    brier_score_loss(test["team1_won"], probabilities), 4
))
print(
    "Learned coefficient:",
    round(
        model.named_steps["logisticregression"].coef_[0][0],
        6,
    ),
)
