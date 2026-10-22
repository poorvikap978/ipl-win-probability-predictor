
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    classification_report,
    brier_score_loss,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
INPUT_FILE = DATA_DIR / "processed" / "match_features.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["match_date"])
df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)


cutoff = pd.Timestamp("2024-01-01")
train = df[df["match_date"] < cutoff].copy()
test = df[df["match_date"] >= cutoff].copy()

numeric_features = [
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

categorical_features = [
    "team1_name",
    "team2_name",
    "venue",

]

features = numeric_features + categorical_features
target = "team1_won"

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

models = {
    "Dummy baseline": DummyClassifier(strategy="prior"),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
    ),
}

print("Training matches:", len(train))
print("Testing matches:", len(test))
print("Train dates:", train["match_date"].min(), "to",
      train["match_date"].max())
print("Test dates:", test["match_date"].min(), "to",
      test["match_date"].max())

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
        
    reversed_probabilities = 1 - probabilities

    print(f"\n===== {name} =====")
    print("Mean predicted probability:", round(probabilities.mean(), 4))
    print("Actual Team 1 win rate:", round(y_test.mean(), 4))
    print("Normal ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
    print("Reversed ROC-AUC:", round(roc_auc_score(y_test, 1 - probabilities), 4)) 
    
    

    print(f"\n===== {name} =====")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
    print("Log loss:", round(log_loss(y_test, probabilities), 4))
    print("Brier score:", round(brier_score_loss(y_test, probabilities), 4))
    print(classification_report(y_test, predictions, zero_division=0))
