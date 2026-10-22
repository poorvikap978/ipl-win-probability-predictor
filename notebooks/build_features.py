
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
INPUT_FILE = DATA_DIR / "processed" / "clean_matches.csv"
OUTPUT_FILE = DATA_DIR / "processed" / "match_features.csv"

# Elo configuration
INITIAL_ELO = 1500
ELO_K = 20

# Load and sort matches
df = pd.read_csv(INPUT_FILE, parse_dates=["match_date"])
df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

# Normalize historical team names
name_map = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Delhi Daredevils": "Delhi Capitals",
    "DelhiCapitals": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
}

for col in [
    "team1_name",
    "team2_name",
    "toss_winner_name",
    "match_winner_name",
]:
    if col in df.columns:
        df[col] = df[col].replace(name_map)

# Toss features
df["team1_won_toss"] = (
    df["team1"] == df["toss_winner"]
).astype(int)

df["toss_decision"] = (
    df["toss_decision"].str.lower().str.strip()
)

# Historical statistics
recent_results = defaultdict(lambda: deque(maxlen=5))
overall_results = defaultdict(lambda: [0, 0])
head_to_head = defaultdict(lambda: [0, 0])
venue_history = defaultdict(lambda: [0, 0])

# Elo ratings
elo_ratings = defaultdict(lambda: INITIAL_ELO)

features = []

# Process one calendar date at a time.
# All matches on a date use history from earlier dates only.
for match_date, day_matches in df.groupby("match_date", sort=True):

    day_matches = day_matches.sort_values("match_id")

    # Save this date's matches and their outcomes for later updates.
    day_updates = []

    # PHASE 1: Generate features without updating any history.
    for row in day_matches.itertuples(index=False):
        team1 = row.team1_name
        team2 = row.team2_name
        venue = row.venue

        form1 = list(recent_results[team1])
        form2 = list(recent_results[team2])

        h2h = head_to_head[(team1, team2)]
        venue1 = venue_history[(team1, venue)]
        venue2 = venue_history[(team2, venue)]

        overall1 = overall_results[team1]
        overall2 = overall_results[team2]

        elo1 = elo_ratings[team1]
        elo2 = elo_ratings[team2]

        features.append({
            "match_id": row.match_id,
            "match_date": row.match_date,
            "season": row.season,
            "team1_name": team1,
            "team2_name": team2,
            "venue": venue,
            "toss_decision": row.toss_decision,
            "team1_won_toss": row.team1_won_toss,

            "team1_elo": elo1,
            "team2_elo": elo2,
            "elo_difference": elo1 - elo2,

            "team1_recent_win_pct": (
                sum(form1) / len(form1) if form1 else 0.5
            ),
            "team2_recent_win_pct": (
                sum(form2) / len(form2) if form2 else 0.5
            ),
            "team1_recent_matches": len(form1),
            "team2_recent_matches": len(form2),

            "team1_overall_win_pct": (
                overall1[0] / overall1[1]
                if overall1[1] else 0.5
            ),
            "team2_overall_win_pct": (
                overall2[0] / overall2[1]
                if overall2[1] else 0.5
            ),

            "team1_h2h_win_pct": (
                h2h[0] / h2h[1] if h2h[1] else 0.5
            ),
            "head_to_head_matches": h2h[1],

            "team1_venue_win_pct": (
                venue1[0] / venue1[1]
                if venue1[1] else 0.5
            ),
            "team1_venue_matches": venue1[1],
            "team2_venue_win_pct": (
                venue2[0] / venue2[1]
                if venue2[1] else 0.5
            ),
            "team2_venue_matches": venue2[1],

            "team1_won": int(row.team1_won),
        })

        won1 = int(row.team1_won)
        won2 = 1 - won1

        # Calculate Elo changes from the ratings at the start of the date.
        expected1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
        expected2 = 1 - expected1

        elo_change1 = ELO_K * (won1 - expected1)
        elo_change2 = ELO_K * (won2 - expected2)

        day_updates.append({
            "team1": team1,
            "team2": team2,
            "venue": venue,
            "won1": won1,
            "won2": won2,
            "elo_change1": elo_change1,
            "elo_change2": elo_change2,
        })

    # PHASE 2: Update histories only after all features for this
    # date have been generated.
    # Accumulate Elo changes before applying them.
    elo_changes = defaultdict(float)

    for update in day_updates:
        team1 = update["team1"]
        team2 = update["team2"]
        venue = update["venue"]
        won1 = update["won1"]
        won2 = update["won2"]

        recent_results[team1].append(won1)
        recent_results[team2].append(won2)

        overall_results[team1][0] += won1
        overall_results[team1][1] += 1
        overall_results[team2][0] += won2
        overall_results[team2][1] += 1

        head_to_head[(team1, team2)][0] += won1
        head_to_head[(team1, team2)][1] += 1
        head_to_head[(team2, team1)][0] += won2
        head_to_head[(team2, team1)][1] += 1

        venue_history[(team1, venue)][0] += won1
        venue_history[(team1, venue)][1] += 1
        venue_history[(team2, venue)][0] += won2
        venue_history[(team2, venue)][1] += 1

        elo_changes[team1] += update["elo_change1"]
        elo_changes[team2] += update["elo_change2"]

    for team, change in elo_changes.items():
        elo_ratings[team] += change

# Save features
features_df = pd.DataFrame(features)
features_df.to_csv(OUTPUT_FILE, index=False)

print("Features created:", features_df.shape)
print("Unique matches:", features_df["match_id"].nunique())
print("Date range:",
      features_df["match_date"].min(),
      "to",
      features_df["match_date"].max())

print("\nColumns:")
print(features_df.columns.tolist())

print("\nElo feature preview:")
print(
    features_df[
        ["team1_name", "team2_name",
         "team1_elo", "team2_elo",
         "elo_difference", "team1_won"]
    ].head(10).to_string(index=False)
)

print("\nDelhi name variants:")
print(
    features_df[
        features_df["team1_name"].str.contains(
            "Delhi", case=False, na=False
        )
        | features_df["team2_name"].str.contains(
            "Delhi", case=False, na=False
        )
    ][["team1_name", "team2_name"]]
    .stack()
    .unique()
)

print("\nSaved to:", OUTPUT_FILE)
