
import pandas as pd
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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

df["form_difference"] = (
    df["team1_recent_win_pct"]
    - df["team2_recent_win_pct"]
)

cutoff = pd.Timestamp("2024-01-01")
train = df[df["match_date"] < cutoff]
test = df[df["match_date"] >= cutoff]

X_train = train[["form_difference"]]
y_train = train["team1_won"]
X_test = test[["form_difference"]]
y_test = test["team1_won"]

models = {
    "Baseline": DummyClassifier(strategy="prior"),
    "Form-difference model": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
    print("Log loss:", round(log_loss(y_test, probabilities), 4))
    print("Brier score:", round(brier_score_loss(y_test, probabilities), 4))
