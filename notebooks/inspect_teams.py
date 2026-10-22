
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

matches = pd.read_csv(DATA_DIR / "ipl_matches_data.csv")

# Inspect supporting team files
for filename in ["teams_data.csv", "team_aliases.csv"]:
    path = DATA_DIR / filename

    print(f"\n{'=' * 15} {filename} {'=' * 15}")

    if path.exists():
        df = pd.read_csv(path)
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print(df.head(15).to_string(index=False))
    else:
        print("File not found:", path)

# Inspect IDs used in the match data
team_columns = ["team1", "team2", "toss_winner", "match_winner"]

for col in team_columns:
    print(f"\n{col}:")
    print("Unique IDs:", sorted(matches[col].dropna().unique()))
    print("Missing:", matches[col].isna().sum())

print("\nMatch result categories:")
print(matches["result"].value_counts(dropna=False))

print("\nMatch types:")
print(matches["match_type"].value_counts(dropna=False))

print("\nSeasons:")
print(matches["season"].value_counts().sort_index())

# Inspect team names in the cleaned dataset
cleaned_file = DATA_DIR / "processed" / "clean_matches.csv"

print(f"\n{'=' * 15} CLEANED TEAM NAMES {'=' * 15}")

if cleaned_file.exists():
    cleaned = pd.read_csv(cleaned_file)

    teams = sorted(
        set(cleaned["team1_name"].dropna()) |
        set(cleaned["team2_name"].dropna())
    )

    for team in teams:
        print(team)

    print("\nTotal unique team names:", len(teams))
else:
    print("Cleaned dataset not found:", cleaned_file)