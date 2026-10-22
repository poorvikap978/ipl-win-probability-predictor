
import joblib
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_DIR / "models" / "recent_form_model.joblib"

saved = joblib.load(MODEL_FILE)
model = saved["model"]

# Example: Team 1's recent win rate is 60%;
# Team 2's recent win rate is 40%.
form_difference = 0.60 - 0.40

X = pd.DataFrame({
    "form_difference": [form_difference]
})

probability = model.predict_proba(X)[0, 1]

print("Form difference:", form_difference)
print("Experimental Team 1 probability:", round(probability, 4))
print("Experimental Team 2 probability:", round(1 - probability, 4))
print("Model type:", saved["model_type"])
