
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
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
df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

cutoff = pd.Timestamp("2024-01-01")
train = df[df["match_date"] < cutoff]
test = df[df["match_date"] >= cutoff]

target = "team1_won"

feature_groups = {
    "A - Toss only": {
        "numeric": ["team1_won_toss"],
        "categorical": ["toss_decision"],
    },
    "B - Toss + recent form": {
        "numeric": [
            "team1_won_toss",
            "team1_recent_win_pct",
            "team2_recent_win_pct",
            "team1_recent_matches",
            "team2_recent_matches",
        ],
        "categorical": ["toss_decision"],
    },
    "C - Toss + overall win rate": {
        "numeric": [
            "team1_won_toss",
            "team1_overall_win_pct",
            "team2_overall_win_pct",
        ],
        "categorical": ["toss_decision"],
    },
    "D - Toss + form + overall win rate": {
        "numeric": [
            "team1_won_toss",
            "team1_recent_win_pct",
            "team2_recent_win_pct",
            "team1_recent_matches",
            "team2_recent_matches",
            "team1_overall_win_pct",
            "team2_overall_win_pct",
        ],
        "categorical": ["toss_decision"],
    },
}

print("Training matches:", len(train))
print("Testing matches:", len(test))
print("Actual test Team 1 win rate:", round(test[target].mean(), 4))

for name, group in feature_groups.items():
    numeric = group["numeric"]
    categorical = group["categorical"]
    columns = numeric + categorical

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000)),
    ])

    model.fit(train[columns], train[target])

    probability = model.predict_proba(test[columns])[:, 1]
    prediction = (probability >= 0.5).astype(int)

    print(f"\n=== {name} ===")
    print("Accuracy:", round(accuracy_score(test[target], prediction), 4))
    print("ROC-AUC:", round(roc_auc_score(test[target], probability), 4))
    print("Log loss:", round(log_loss(test[target], probability), 4))
    print("Brier score:", round(brier_score_loss(test[target], probability), 4))
