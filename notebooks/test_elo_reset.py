
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "processed" / "clean_matches.csv"

INITIAL_ELO = 1500
ELO_K = 20
SEASON_RESET = 0.25

df = pd.read_csv(DATA_FILE, parse_dates=["match_date"])
df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

# Keep the same chronological train/test split.
cutoff = pd.Timestamp("2024-01-01")
test_start_index = df.index[df["match_date"] >= cutoff][0]

ratings = {}
current_season = None
test_records = []

for row in df.itertuples(index=False):
    season = row.season
    team1 = row.team1_name
    team2 = row.team2_name

    # Reset partway toward 1500 when the season changes.
    if season != current_season:
        for team in ratings:
            ratings[team] = (
                (1 - SEASON_RESET) * ratings[team]
                + SEASON_RESET * INITIAL_ELO
            )
        current_season = season

    elo1 = ratings.get(team1, INITIAL_ELO)
    elo2 = ratings.get(team2, INITIAL_ELO)

    # Evaluate only matches in the held-out test period.
    if row.match_date >= cutoff:
        probability = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
        test_records.append({
            "season": season,
            "team1_won": int(row.team1_won),
            "elo_difference": elo1 - elo2,
            "probability": probability,
        })

    # Update ratings only after recording the pre-match prediction.
    won1 = int(row.team1_won)
    won2 = 1 - won1

    expected1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    expected2 = 1 - expected1

    ratings[team1] = elo1 + ELO_K * (won1 - expected1)
    ratings[team2] = elo2 + ELO_K * (won2 - expected2)

results = pd.DataFrame(test_records)
y = results["team1_won"]
p = results["probability"]
pred = (p >= 0.5).astype(int)

print("===== Season-reset Elo: Overall =====")
print("Test matches:", len(results))
print("Accuracy:", round(accuracy_score(y, pred), 4))
print("ROC-AUC:", round(roc_auc_score(y, p), 4))
print("Log loss:", round(log_loss(y, p, labels=[0, 1]), 4))
print("Brier score:", round(brier_score_loss(y, p), 4))

print("\n===== Season-by-season results =====")
for season, group in results.groupby("season", sort=True):
    y_season = group["team1_won"]
    p_season = group["probability"]
    pred_season = (p_season >= 0.5).astype(int)

    print(f"\nSeason {season} | Matches: {len(group)}")
    print("Accuracy:", round(accuracy_score(y_season, pred_season), 4))
    print("Log loss:", round(log_loss(y_season, p_season, labels=[0, 1]), 4))
    print("Brier score:", round(brier_score_loss(y_season, p_season), 4))

    if y_season.nunique() == 2:
        print("ROC-AUC:", round(roc_auc_score(y_season, p_season), 4))
    else:
        print("ROC-AUC: unavailable")
