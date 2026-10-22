
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)

# Load the existing pre-match features
df = pd.read_csv("data/processed/match_features.csv")
df["match_date"] = pd.to_datetime(df["match_date"])

# Use the same chronological test period as the existing model
test = df[df["match_date"] >= "2024-01-01"].copy()

y_true = test["team1_won"].astype(int)

# Baseline 1: constant 50% probability
p_50 = [0.5] * len(test)

# Baseline 2: constant probability based on training-set win rate
train = df[df["match_date"] < "2024-01-01"]
training_win_rate = train["team1_won"].mean()
p_prior = [training_win_rate] * len(test)

# Baseline 3: toss-only probability
# Estimate Team 1's win rate separately for each toss outcome.
toss_rates = train.groupby("team1_won_toss")["team1_won"].mean()
overall_rate = training_win_rate

p_toss = (
    test["team1_won_toss"]
    .map(toss_rates)
    .fillna(overall_rate)
    .clip(0.01, 0.99)
    .to_numpy()
)

baselines = {
    "50% baseline": p_50,
    "Training win-rate baseline": p_prior,
    "Toss-only baseline": p_toss,
}

print(f"Training matches: {len(train)}")
print(f"Test matches: {len(test)}")
print(f"Actual Team 1 win rate in test: {y_true.mean():.3f}")
print(f"Training Team 1 win rate: {training_win_rate:.3f}")

print("\nBASELINE COMPARISON")
print("-" * 78)

for name, probabilities in baselines.items():
    p = pd.Series(probabilities, index=test.index).to_numpy()
    predicted = (p >= 0.5).astype(int)

    print(f"\n{name}")
    print(f"  Accuracy:  {accuracy_score(y_true, predicted):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_true, p):.4f}")
    print(f"  Log loss:  {log_loss(y_true, p, labels=[0, 1]):.4f}")
    print(f"  Brier:     {brier_score_loss(y_true, p):.4f}")
