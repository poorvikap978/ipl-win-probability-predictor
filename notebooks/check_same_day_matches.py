
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
FILE = PROJECT_DIR / "data" / "processed" / "clean_matches.csv"

df = pd.read_csv(FILE, parse_dates=["match_date"])
df = df.sort_values(["match_date", "match_id"])

# Show dates containing multiple matches.
counts = df.groupby("match_date").size()
multi_dates = counts[counts > 1].index

sample = df[df["match_date"].isin(multi_dates)].copy()

print("Dates with multiple matches:", len(multi_dates))
print("\nFirst 30 matches on those dates:")
print(
    sample[
        [
            "match_date",
            "match_id",
            "team1_name",
            "team2_name",
            "match_winner_name",
        ]
    ].head(30).to_string(index=False)
)

print("\nCheck whether the source contains a match-time column:")
print([col for col in df.columns if "time" in col.lower()])
