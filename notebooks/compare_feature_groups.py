
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/match_features.csv")
df["match_date"] = pd.to_datetime(df["match_date"])
df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

train = df[df["match_date"] < "2024-01-01"].copy()
test = df[df["match_date"] >= "2024-01-01"].copy()

y_train = train["team1_won"].astype(int)
y_test = test["team1_won"].astype(int)

feature_groups = {
    "Toss only": [
        "team1_won_toss",
    ],
    "Elo only": [
        "team1_elo",
        "team2_elo",
        "elo_difference",
    ],
    "Recent form": [
        "team1_recent_win_pct",
        "team2_recent_win_pct",
        "team1_recent_matches",
        "team2_recent_matches",
    ],
    "Overall strength": [
        "team1_overall_win_pct",
        "team2_overall_win_pct",
    ],
    "Head-to-head": [
        "team1_h2h_win_pct",
        "head_to_head_matches",
    ],
    "Venue history": [
        "team1_venue_win_pct",
        "team1_venue_matches",
        "team2_venue_win_pct",
        "team2_venue_matches",
    ],
}

print(f"Training matches: {len(train)}")
print(f"Test matches: {len(test)}")
print("\nFEATURE-GROUP COMPARISON")

for name, features in feature_groups.items():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, features),
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=2000)),
    ])

    model.fit(train[features], y_train)
    probabilities = model.predict_proba(test[features])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(f"\n{name}")
    print(f"  Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"  ROC-AUC:  {roc_auc_score(y_test, probabilities):.4f}")
    print(f"  Log loss: {log_loss(y_test, probabilities):.4f}")
    print(f"  Brier:    {brier_score_loss(y_test, probabilities):.4f}")
