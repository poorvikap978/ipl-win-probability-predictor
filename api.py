
import fastapi

from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import predictor
from fastapi import FastAPI

app = FastAPI(
    title="IPL Win Probability Predictor",
    description="An API for predicting IPL match outcomes.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "IPL Win Probability Predictor API is running!",
        "status": "success",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}

class PredictionRequest(BaseModel):
    team1: str
    team2: str
    match_date: str





@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        return predictor.predict_winner(
            request.team1,
            request.team2,
            request.match_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
