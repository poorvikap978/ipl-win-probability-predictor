# 🏏 IPL Win Probability Predictor

A machine learning project that analyzes historical Indian Premier League (IPL) cricket data to predict match outcomes and display experimental win-probability estimates through a simple web interface.

## Features

- Predicts a likely winner based on historical team performance.
- Analyzes recent team form and head-to-head results.
- Displays experimental win-probability estimates.
- Provides a FastAPI backend for predictions.
- Includes a web interface built with HTML, CSS, and JavaScript.
- Includes data-processing, feature-engineering, and model-evaluation scripts.

## Tech Stack

- **Python** — data processing and prediction logic
- **Pandas** — data analysis
- **Scikit-learn** — machine learning
- **Joblib** — loading the saved model
- **FastAPI** — REST API
- **HTML, CSS, JavaScript** — frontend

## Project Structure

```text
ipl-predictor/
├── api.py
├── predictor.py
├── requirements.txt
├── data/
│   └── processed/
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── models/
│   └── recent_form_model.joblib
└── notebooks/
```

## Setup and Run

### 1. Clone the repository

```bash
git clone https://github.com/poorvikap978/ipl-win-probability-predictor.git
cd ipl-win-probability-predictor
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API

```bash
python3 -m uvicorn api:app --host 127.0.0.1 --port 8000
```

### 5. Open the frontend

Open `frontend/index.html` using VS Code Live Server while the API is running.

The API documentation is available at:

`http://127.0.0.1:8000/docs`

## Model Limitations

The win-probability estimates are experimental and have limited validation. They should not be treated as reliable real-world forecasts or betting advice. Actual match outcomes can depend on many factors not represented in the model.

## Future Improvements

- Improve model evaluation and probability calibration.
- Explore additional pre-match features.
- Expand automated testing.
- Deploy the API and frontend online.

## Author

Poorvika
