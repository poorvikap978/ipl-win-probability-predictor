
import pandas as pd
from pathlib import Path

path = Path("data/processed/clean_matches.csv")
df = pd.read_csv(path, parse_dates=["match_date"])

prediction_date = pd.Timestamp("2026-04-01")

history = df[df["match_date"] < prediction_date]

for team in [
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
]:
    rows = history[
        (history["team1_name"] == team)
        | (history["team2_name"] == team)
    ].sort_values(["match_date", "match_id"])

    print(f"\n{team}")
    print("Historical matches:", len(rows))
    print("Latest match used:", rows["match_date"].max())
    print("Matches on/after prediction date:",
          (rows["match_date"] >= prediction_date).sum())
