# app/models.py
from pydantic import BaseModel, Field
from typing import List, Optional

class PredictRequest(BaseModel):
    crop: str
    soil: str
    location: str
    date: str  # ISO date string e.g. "2025-09-01"

class DayData(BaseModel):
    day: int
    groundwater: float
    rainfall: float

class PredictResponse(BaseModel):
    days: List[DayData]

class CalendarDayResponse(BaseModel):
    date: str
    suggestedCrop: Optional[str]
    requiredWaterPct: float
