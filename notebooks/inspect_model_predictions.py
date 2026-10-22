
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

PROJECT_DIR = Path(__file__).resolve().parent.parent
FILE = PROJECT_DIR / "data" / "processed" / "match_features.csv"

df = pd.read_csv(FILE, parse_dates=["match_date"])
df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

cutoff = pd.Timestamp("2024-01-01")
train = df[df["match_date"] < cutoff].copy()
test = df[df["match_date"] >= cutoff].copy()

numeric = [
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
]

categorical = ["team1_name", "team2_name", "venue"]
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

model.fit(train[columns], train["team1_won"])

prob = model.predict_proba(test[columns])[:, 1]
actual = test["team1_won"].to_numpy()

print("Actual Team 1 win rate:", round(actual.mean(), 4))
print("Mean predicted Team 1 win probability:", round(prob.mean(), 4))
print("ROC-AUC:", round(roc_auc_score(actual, prob), 4))
print("ROC-AUC if probabilities are reversed:",
      round(roc_auc_score(actual, 1 - prob), 4))
print("Log loss:", round(log_loss(actual, prob), 4))
print("Brier score:", round(brier_score_loss(actual, prob), 4))

print("\nAverage Elo difference by outcome:")
print(test.groupby("team1_won")["elo_difference"].agg(
    ["count", "mean", "median"]
))

print("\nHistorical feature ranges:")
print(df[numeric].describe().T[
    ["mean", "std", "min", "max"]
].round(3))
