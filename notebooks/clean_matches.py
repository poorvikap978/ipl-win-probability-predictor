
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


matches = pd.read_csv(DATA_DIR / "ipl_matches_data.csv")
deliveries = pd.read_csv(DATA_DIR / "ball_by_ball_data.csv")
teams = pd.read_csv(DATA_DIR / "teams_data.csv")

print("Original matches:", len(matches))


teams["team_id"] = pd.to_numeric(
    teams["team_id"], errors="coerce"
)
teams = teams.dropna(subset=["team_id"])
teams["team_id"] = teams["team_id"].astype(int)


if teams["team_id"].duplicated().any():
    raise ValueError("Duplicate team IDs in teams_data.csv")

team_map = teams.set_index("team_id")["team_name"].to_dict()

name_map = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Delhi Daredevils": "Delhi Capitals",
    "DelhiCapitals": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",

}

team_map = {
    team_id: name_map.get(team_name, team_name)
    for team_id, team_name in team_map.items()
}

team_columns = [
    "team1", "team2", "toss_winner", "match_winner"
]

for col in team_columns:
    matches[col] = pd.to_numeric(matches[col], errors="coerce")


used_ids = set()

for col in team_columns:
    used_ids.update(matches[col].dropna().astype(int).unique())

unmapped_ids = sorted(used_ids - set(team_map))

if unmapped_ids:
    print("WARNING: Unmapped team IDs:", unmapped_ids)


for col in team_columns:
    matches[col] = matches[col].astype("Int64")
    matches[col.replace("_", "_") + "_name"] = (
        matches[col].map(team_map)
    )


print("\nDuplicate match IDs:", matches["match_id"].duplicated().sum())

if matches["match_id"].duplicated().any():
    raise ValueError("Duplicate match IDs found; investigate first.")

matches["match_date"] = pd.to_datetime(
    matches["match_date"], errors="coerce"
)

print("Invalid dates:", matches["match_date"].isna().sum())

invalid_teams = (
    matches["team1"].isna()
    | matches["team2"].isna()
    | (matches["team1"] == matches["team2"])
)

print("Rows with invalid team assignments:", invalid_teams.sum())


invalid_toss = ~matches["toss_winner"].isin(
    matches[["team1", "team2"]].stack().unique()
)


invalid_toss = ~(
    (matches["toss_winner"] == matches["team1"])
    | (matches["toss_winner"] == matches["team2"])
)

print("Rows with toss winner outside participating teams:",
      invalid_toss.sum())

invalid_winner = ~(
    (matches["match_winner"] == matches["team1"])
    | (matches["match_winner"] == matches["team2"])
)

print("Rows with winner outside participating teams:",
      invalid_winner.sum())


super_over_matches = set(
    deliveries.loc[
        deliveries["is_super_over"].eq(True), "match_id"
    ].unique()
)

matches["had_super_over"] = matches["match_id"].isin(
    super_over_matches
)

print("Matches with Super Over:", matches["had_super_over"].sum())


matches["result"] = matches["result"].str.strip().str.lower()

print("\nResult categories:")
print(matches["result"].value_counts(dropna=False))


eligible = (
    matches["result"].eq("win")
    & ~matches["had_super_over"]
    & ~invalid_teams
    & ~invalid_toss
    & ~invalid_winner
    & matches["match_date"].notna()
    & matches["team1_name"].notna()
    & matches["team2_name"].notna()
    & matches["toss_winner_name"].notna()
    & matches["match_winner_name"].notna()
)

clean_matches = matches.loc[eligible].copy()


clean_matches["team1_won"] = (
    clean_matches["match_winner"]
    == clean_matches["team1"]
).astype(int)


clean_matches = clean_matches.sort_values(
    ["match_date", "match_id"]
).reset_index(drop=True)

output_file = OUTPUT_DIR / "clean_matches.csv"
clean_matches.to_csv(output_file, index=False)

print("\n========== CLEANING SUMMARY ==========")
print("Original matches:", len(matches))
print("Eligible decisive matches:", len(clean_matches))
print("Excluded matches:", len(matches) - len(clean_matches))
print("\nTarget distribution:")
print(clean_matches["team1_won"].value_counts())
print("\nTeam names:")
print(sorted(
    set(clean_matches["team1_name"])
    | set(clean_matches["team2_name"])
))
print("\nDate range:")
print(clean_matches["match_date"].min())
print("to", clean_matches["match_date"].max())
print("\nSaved to:", output_file)
