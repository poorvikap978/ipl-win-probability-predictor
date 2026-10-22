
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
    "Toss only": {
        "numeric": ["team1_won_toss"],
        "categorical": ["toss_decision"],
    },
    "Historical features": {
        "numeric": [
            "team1_recent_win_pct",
            "team2_recent_win_pct",
            "team1_recent_matches",
            "team2_recent_matches",
            "team1_overall_win_pct",
            "team2_overall_win_pct",
            "team1_h2h_win_pct",
            "head_to_head_matches",
            "team1_venue_win_pct",
            "team1_venue_matches",
            "team2_venue_win_pct",
            "team2_venue_matches",
            "team1_elo",
            "team2_elo",
            "elo_difference",
        ],
        "categorical": [
            "team1_name",
            "team2_name",
            "venue",
        ],
    },
}

print("Training matches:", len(train))
print("Testing matches:", len(test))
print("Test Team 1 win rate:", round(test[target].mean(), 4))

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

    probabilities = model.predict_proba(test[columns])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(f"\n=== {name} ===")
    print("Accuracy:", round(accuracy_score(test[target], predictions), 4))
    print("ROC-AUC:", round(roc_auc_score(test[target], probabilities), 4))
    print("Log loss:", round(log_loss(test[target], probabilities), 4))
    print("Brier score:", round(brier_score_loss(test[target], probabilities), 4))
