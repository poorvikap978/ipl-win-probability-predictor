
import pandas as pd
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = (
    PROJECT_DIR / "data" / "processed" / "match_features.csv"
)
MODEL_DIR = PROJECT_DIR / "models"
MODEL_FILE = MODEL_DIR / "recent_form_model.joblib"

df = pd.read_csv(DATA_FILE, parse_dates=["match_date"])
df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

df["form_difference"] = (
    df["team1_recent_win_pct"]
    - df["team2_recent_win_pct"]
)

cutoff = pd.Timestamp("2024-01-01")
train = df[df["match_date"] < cutoff].copy()
test = df[df["match_date"] >= cutoff].copy()

feature = ["form_difference"]

X_train = train[feature]
y_train = train["team1_won"]
X_test = test[feature]
y_test = test["team1_won"]

baseline = DummyClassifier(strategy="prior")
baseline.fit(X_train, y_train)
baseline_probabilities = baseline.predict_proba(X_test)[:, 1]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000)),
])

model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)[:, 1]

print("Training matches:", len(train))
print("Testing matches:", len(test))

for name, probs in [
    ("Baseline", baseline_probabilities),
    ("Recent-form model", probabilities),
]:
    print(f"\n{name}")
    print("ROC-AUC:", round(roc_auc_score(y_test, probs), 4))
    print("Log loss:", round(log_loss(y_test, probs), 4))
    print("Brier score:", round(brier_score_loss(y_test, probs), 4))

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": feature,
        "training_cutoff": "2024-01-01",
        "model_type": "experimental_recent_form_difference",
    },
    MODEL_FILE,
)

print("\nSaved model to:", MODEL_FILE)
