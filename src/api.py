"""
FastAPI model-serving app.

Endpoints:
  GET  /health           – liveness + which model is loaded
  POST /predict          – batch prediction (1..N applications), persists rows
  GET  /predictions      – past predictions (paginated), newest first

FastAPI features used: Pydantic request/response models, response_model,
status codes, model loading once at startup via lifespan, HTTPException error
handling, and DB session dependency injection with Depends().
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db import IngestionStat, Prediction, get_db, init_db  # noqa: F401
from src.model_service import ModelService
from src.schemas import PastPrediction, PredictionRequest, PredictionResponse

# Populated once at startup.
ml_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        ml_models["service"] = ModelService.load()
    except FileNotFoundError as exc:
        # Fail fast with a clear message rather than crashing on first request.
        raise RuntimeError(str(exc)) from exc
    yield
    ml_models.clear()


app = FastAPI(title="Loan Approval Prediction API", version="1.0.0", lifespan=lifespan)


def get_model() -> ModelService:
    service = ml_models.get("service")
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )
    return service


@app.get("/health")
def health(model: ModelService = Depends(get_model)):
    return {"status": "ok", "model_name": model.model_name, "backend": model.backend}


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict(
    request: PredictionRequest,
    model: ModelService = Depends(get_model),
    db: Session = Depends(get_db),
):
    rows = [app_.model_dump() for app_ in request.applications]
    df = pd.DataFrame(rows)
    try:
        results = model.predict(df)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prediction failed: {exc}",
        ) from exc

    for row, res in zip(rows, results):
        db.add(
            Prediction(
                source=request.source,
                prediction=res["prediction"],
                probability=res["probability"],
                cibil_score=int(row["cibil_score"]),
                loan_amount=float(row["loan_amount"]),
                income_annum=float(row["income_annum"]),
                features=row,
            )
        )
    db.commit()

    return PredictionResponse(model_name=model.model_name, results=results)


@app.get("/predictions", response_model=list[PastPrediction])
def past_predictions(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    source: str | None = Query(None),
    db: Session = Depends(get_db),
):
    stmt = select(Prediction).order_by(Prediction.created_at.desc())
    if source:
        stmt = stmt.where(Prediction.source == source)
    stmt = stmt.limit(limit).offset(offset)
    return db.execute(stmt).scalars().all()
