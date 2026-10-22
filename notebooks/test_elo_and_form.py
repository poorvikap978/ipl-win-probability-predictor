
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

y_train = train["team1_won"]
y_test = test["team1_won"]

feature_sets = {
    "Recent form": ["form_difference"],
    "Elo + recent form": [
        "elo_difference",
        "form_difference",
    ],
}

baseline = DummyClassifier(strategy="prior")
baseline.fit(train[["form_difference"]], y_train)

models = {
    "Baseline": (
        baseline,
        test[["form_difference"]],
    )
}

for name, features in feature_sets.items():
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )
    model.fit(train[features], y_train)
    models[name] = (model, test[features])

print("Training matches:", len(train))
print("Test matches:", len(test))

for name, (model, X_test) in models.items():
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(f"\n{name}")
    print("Accuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))
    print("ROC-AUC:", round(
        roc_auc_score(y_test, probabilities), 4
    ))
    print("Log loss:", round(
        log_loss(y_test, probabilities), 4
    ))
    print("Brier score:", round(
        brier_score_loss(y_test, probabilities), 4
    ))
