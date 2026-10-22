
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "clean_matches.csv"
MODEL_PATH = BASE_DIR / "models" / "recent_form_model.joblib"

NAME_MAP = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Delhi Daredevils": "Delhi Capitals",
    "DelhiCapitals": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
}


def load_experimental_model():
    """Load the saved experimental model, if available."""
    if not MODEL_PATH.exists():
        return None

    try:
        saved_model = joblib.load(MODEL_PATH)

        if (
            saved_model.get("features") != ["form_difference"]
            or "model" not in saved_model
        ):
            return None

        return saved_model["model"]

    except (OSError, ValueError, KeyError, AttributeError):
        return None


def predict_winner(team1: str, team2: str, match_date: str) -> dict:
    if team1.strip().casefold() == team2.strip().casefold():
        raise ValueError("Please select two different teams.")

    try:
        prediction_date = pd.to_datetime(
            match_date, format="%Y-%m-%d", errors="raise"
        ).normalize()
    except (ValueError, TypeError):
        raise ValueError("Use a valid date in YYYY-MM-DD format.")

    if not DATA_PATH.exists():
        raise FileNotFoundError("Cleaned historical matches file not found.")

    df = pd.read_csv(DATA_PATH, parse_dates=["match_date"])

    for col in ["team1_name", "team2_name"]:
        df[col] = df[col].replace(NAME_MAP)

    teams = set(df["team1_name"].dropna())
    teams.update(df["team2_name"].dropna())

    def find_team(name):
        return next(
            (
                team for team in teams
                if team.casefold() == name.strip().casefold()
            ),
            None,
        )

    team1_name = find_team(team1)
    team2_name = find_team(team2)

    if not team1_name or not team2_name:
        raise ValueError("Unknown team. Check the team name and try again.")

    # Only use matches played strictly before the requested date.
    history = df[df["match_date"] < prediction_date].copy()

    if history.empty:
        raise ValueError("No historical matches are available before this date.")

    def team_stats(team):
        rows = history[
            (history["team1_name"] == team)
            | (history["team2_name"] == team)
        ].sort_values(["match_date", "match_id"])

        if rows.empty:
            raise ValueError(f"No previous matches found for {team}.")

        results = []

        for row in rows.itertuples(index=False):
            if row.team1_name == team:
                results.append(int(row.team1_won))
            else:
                results.append(1 - int(row.team1_won))

        last_five = results[-5:]

        return {
            "recent_win_pct": sum(last_five) / len(last_five),
            "recent_matches": len(last_five),
            "overall_win_pct": sum(results) / len(results),
            "matches_played": len(results),
        }

    stats1 = team_stats(team1_name)
    stats2 = team_stats(team2_name)

    # Historical meetings between the two selected teams.
    h2h = history[
        (
            (history["team1_name"] == team1_name)
            & (history["team2_name"] == team2_name)
        )
        | (
            (history["team1_name"] == team2_name)
            & (history["team2_name"] == team1_name)
        )
    ].sort_values(["match_date", "match_id"])

    team1_h2h_wins = 0

    for row in h2h.itertuples(index=False):
        if row.team1_name == team1_name:
            team1_h2h_wins += int(row.team1_won)
        else:
            team1_h2h_wins += 1 - int(row.team1_won)

    h2h_matches = len(h2h)
    h2h_pct = (
        team1_h2h_wins / h2h_matches if h2h_matches else None
    )

    # Existing rule-based winner prediction remains unchanged.
    if stats1["recent_win_pct"] != stats2["recent_win_pct"]:
        winner = (
            team1_name
            if stats1["recent_win_pct"] > stats2["recent_win_pct"]
            else team2_name
        )
        reason = "Higher recent win percentage across the last five matches."

    elif stats1["overall_win_pct"] != stats2["overall_win_pct"]:
        winner = (
            team1_name
            if stats1["overall_win_pct"] > stats2["overall_win_pct"]
            else team2_name
        )
        reason = (
            "Recent form is tied; overall historical win rate "
            "breaks the tie."
        )

    elif h2h_matches and h2h_pct != 0.5:
        winner = team1_name if h2h_pct > 0.5 else team2_name
        reason = (
            "Recent form and overall rate are tied; "
            "head-to-head breaks the tie."
        )

    else:
        winner = None
        reason = "Available historical signals are tied."

    # Experimental ML probabilities; these do not determine the winner above.
    experimental_probabilities = None
    experimental_model = load_experimental_model()

    if experimental_model is not None:
        try:
            form_difference = (
                stats1["recent_win_pct"] - stats2["recent_win_pct"]
            )

            model_input = pd.DataFrame({
                "form_difference": [form_difference]
            })

            probabilities = experimental_model.predict_proba(model_input)[0]
            classes = list(experimental_model.classes_)
            team1_class_index = classes.index(1)

            team1_probability = float(probabilities[team1_class_index])
            team2_probability = 1.0 - team1_probability

            experimental_probabilities = {
                "team1": round(team1_probability, 4),
                "team2": round(team2_probability, 4),
                "team1_label": team1_name,
                "team2_label": team2_name,
                "model_type": "experimental_recent_form_difference",
                "disclaimer": (
                    "Experimental estimates only. Model performance is "
                    "limited, and these probabilities have not been "
                    "validated or calibrated for reliable real-world use."
                ),
            }

        except (ValueError, KeyError, AttributeError, IndexError):
            # A model problem should not break the existing prediction.
            experimental_probabilities = None

    return {
        "match_date": prediction_date.strftime("%Y-%m-%d"),
        "team1": team1_name,
        "team2": team2_name,
        "team1_recent_win_pct": round(stats1["recent_win_pct"], 3),
        "team2_recent_win_pct": round(stats2["recent_win_pct"], 3),
        "team1_recent_matches": stats1["recent_matches"],
        "team2_recent_matches": stats2["recent_matches"],
        "team1_overall_win_pct": round(stats1["overall_win_pct"], 3),
        "team2_overall_win_pct": round(stats2["overall_win_pct"], 3),
        "team1_matches_played": stats1["matches_played"],
        "team2_matches_played": stats2["matches_played"],
        "head_to_head_matches": h2h_matches,
        "team1_head_to_head_wins": team1_h2h_wins,
        "team2_head_to_head_wins": h2h_matches - team1_h2h_wins,
        "team1_head_to_head_win_pct": (
            round(h2h_pct, 3) if h2h_pct is not None else None
        ),
        "predicted_winner": winner,
        "method": "historical form, overall rate, and head-to-head rule",
        "probability_estimates_available": False,
        "experimental_probabilities": experimental_probabilities,
        "explanation": reason,
        "historical_cutoff": "Only matches before the requested date",
    }
