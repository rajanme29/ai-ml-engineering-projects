#import necessary modules
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pathlib import Path
import time
import sys


#dynamically locate the project root folder
PROJ_ROOT = Path(__file__).resolve().parents[1]

#source code path
SRC_PATH = PROJ_ROOT / "src"

#check whether the source code path is in the python path
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from predict import SentimentPredictor

#initialize Fastapi app with following metadata /docs
title = "Flipkart Sentiment Classification API"
description = "REST API for classifying customer feedback sentiment into positive, negative, neutral or unknown"

app = FastAPI(
    title=title,
    description=description,
    version="1.0.0"
)

#load the predictor model
predictor = SentimentPredictor()

#request schema
class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Feedback text to classify"
    )

# response schema
class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    latency_ms: float

#endpoint landing page
@app.get("/")
def root():
    return {
        "message": "Customer Feedback Sentiment Classification API",
        "version": "1.0.0",
        "status" : "running"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy and ready for sentiment prediction",
        "model_loaded": predictor.model is not None
    }

# Sentiment Prediction Endpoint
#accepts the customer feedback to predict the sentiment
@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: PredictionRequest):
    try:
        #start latency measurement
        start_time = time.perf_counter()

        # perform sentiment prediction
        prediction = predictor.predict_sentiment(
            request.text
        )

        # stop latency measurement
        end_time = time.perf_counter()

        ## Calculate inference latency in milliseconds
        latency_ms = (end_time - start_time) * 1000

        return {
            "sentiment": prediction["sentiment"],
            "confidence": prediction["confidence"],
            "latency_ms": round(latency_ms, 3)
        }

    # Invalid user input
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    # Unexpected inference failure
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(error)}"
        )