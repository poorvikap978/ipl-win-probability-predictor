
import pandas as pd

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
file_path = PROJECT_DIR / "data" / "processed" / "clean_matches.csv"

df = pd.read_csv(file_path)

# Reconstruct the target independently from the winner ID.
expected_target = (df["match_winner"] == df["team1"]).astype(int)

mismatches = df["team1_won"] != expected_target

print("Total matches:", len(df))
print("Target mismatches:", int(mismatches.sum()))

if mismatches.any():
    print("\nExamples of mismatches:")
    print(
        df.loc[
            mismatches,
            [
                "match_id",
                "team1",
                "team2",
                "match_winner",
                "team1_won",
            ],
        ].head(10).to_string(index=False)
    )
else:
    print("All target labels match the recorded winner.")
