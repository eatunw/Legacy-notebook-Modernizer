"""
main.py — FastAPI application exposing the Titanic survival prediction pipeline.

Fixes addressed
---------------
ISSUE-04 : Prediction uses the serialised Pipeline (encoder + model), so encoding
           is always consistent between training and inference.
ISSUE-05 : Pydantic validation (schema.py) rejects malformed input before it
           reaches the model; HTTPException is raised for operational errors.
ISSUE-06 : Business logic lives in model.py; this file handles only HTTP concerns.
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, status

from app.schema import PredictRequest, PredictResponse
from app.model import load_pipeline, predict, FEATURES

# ── module-level pipeline handle (populated at startup) ──────────────────────
_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the serialised pipeline once at startup; release on shutdown."""
    global _pipeline
    try:
        _pipeline = load_pipeline()
    except FileNotFoundError as exc:
        # Surface a clear message in the server log; requests will return 503.
        print(f"[STARTUP WARNING] {exc}")
        _pipeline = None
    yield
    _pipeline = None


app = FastAPI(
    title="Titanic Survival Predictor",
    description="Production refactor of the Titanic Kaggle tutorial notebook.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict passenger survival",
    responses={
        422: {"description": "Validation error — malformed request body"},
        503: {"description": "Model pipeline not loaded"},
    },
)
def predict_survival(request: PredictRequest) -> PredictResponse:
    """
    Accept the four original notebook features and return a survival prediction.

    - **Pclass**: passenger class (1, 2, or 3)
    - **Sex**: 'male' or 'female'
    - **SibSp**: number of siblings / spouses aboard
    - **Parch**: number of parents / children aboard
    """
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not available. Run app/model.py to train and serialise the pipeline.",
        )

    # Build a single-row DataFrame in the exact column order the pipeline expects.
    row = pd.DataFrame(
        [
            {
                "Pclass": request.Pclass,
                "Sex": request.Sex,
                "SibSp": request.SibSp,
                "Parch": request.Parch,
            }
        ]
    )[FEATURES]

    try:
        labels, probas = predict(_pipeline, row)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return PredictResponse(
        survived=int(labels[0]),
        probability=float(round(probas[0], 6)),
    )
