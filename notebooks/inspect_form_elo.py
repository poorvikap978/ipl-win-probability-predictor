
import pandas as pd
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

PROJECT_DIR = Path(__file__).resolve().parent.parent
FILE = PROJECT_DIR / "data" / "processed" / "match_features.csv"

df = pd.read_csv(FILE, parse_dates=["match_date"])
df = df.sort_values(["match_date", "match_id"])

df["form_difference"] = (
    df["team1_recent_win_pct"]
    - df["team2_recent_win_pct"]
)

train = df[df["match_date"] < "2024-01-01"]

feature_sets = {
    "Recent form": ["form_difference"],
    "Elo + recent form": [
        "elo_difference",
        "form_difference",
    ],
}

for name, features in feature_sets.items():
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )

    model.fit(train[features], train["team1_won"])

    coefficients = model.named_steps[
        "logisticregression"
    ].coef_[0]

    print(f"\n{name}")
    for feature, coefficient in zip(features, coefficients):
        print(f"{feature}: {coefficient:.6f}")
