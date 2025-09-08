# app/routers/predict.py
from fastapi import APIRouter, HTTPException
from ..models import PredictRequest, PredictResponse, DayData
from ..services.ml_service import ml_service
from datetime import datetime

router = APIRouter(prefix="/predict", tags=["predict"])

@router.post("/", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        # Validate date parse
        _ = datetime.fromisoformat(req.date)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use ISO date like 2025-09-01.")
    days = ml_service.predict_month_daily(req.crop, req.soil, req.location, req.date)
    return {"days": [DayData(**d) for d in days]}
