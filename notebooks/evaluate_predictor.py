
from pathlib import Path
import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "processed"
    / "clean_matches.csv"
)

df = pd.read_csv(DATA_PATH, parse_dates=["match_date"])

NAME_MAP = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Delhi Daredevils": "Delhi Capitals",
    "DelhiCapitals": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
}

for col in ["team1_name", "team2_name"]:
    df[col] = df[col].replace(NAME_MAP)

df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

# Evaluate only the most recent 20% of matches.
split_index = int(len(df) * 0.8)
test_df = df.iloc[split_index:].copy()
history = df.iloc[:split_index].copy()

correct = 0
total = 0
skipped = 0

for row in test_df.itertuples(index=False):
    # Do not use matches from the same date as historical information.
    past = history[history["match_date"] < row.match_date]

    if past.empty:
        skipped += 1
        continue

    def recent_win_pct(team):
        matches = past[
            (past["team1_name"] == team)
            | (past["team2_name"] == team)
        ].sort_values(["match_date", "match_id"])

        if matches.empty:
            return None

        results = []
        for match in matches.itertuples(index=False):
            if match.team1_name == team:
                results.append(int(match.team1_won))
            else:
                results.append(1 - int(match.team1_won))

        return sum(results[-5:]) / len(results[-5:])

    pct1 = recent_win_pct(row.team1_name)
    pct2 = recent_win_pct(row.team2_name)

    if pct1 is None or pct2 is None or pct1 == pct2:
        skipped += 1
        continue

    predicted_team1_wins = pct1 > pct2
    actual_team1_wins = bool(row.team1_won)

    if predicted_team1_wins == actual_team1_wins:
        correct += 1

    total += 1

print("IPL Predictor — Historical Evaluation")
print("-------------------------------------")
print(f"Test matches evaluated: {total}")
print(f"Matches skipped: {skipped}")

if total:
    print(f"Accuracy: {correct / total:.3f} ({correct / total:.1%})")
else:
    print("No eligible matches were evaluated.")
