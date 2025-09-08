# app/routers/analytics.py
from fastapi import APIRouter, Query, HTTPException
from typing import List
from ..models import DayData, CalendarDayResponse
from ..services.ml_service import ml_service
from ..services.data_service import suggested_day_info
from datetime import datetime
from calendar import monthrange

router = APIRouter(prefix="/analytics", tags=["analytics"])

@router.get("/monthly", response_model=List[DayData])
def monthly(year: int = Query(...), month: int = Query(...), crop: str = Query("Maize"), soil: str = Query("Loamy"), location: str = Query("Your Farm")):
    """
    Returns daily groundwater/rainfall for a given year & month.
    month: 1-12
    """
    if not (1 <= month <= 12):
        raise HTTPException(status_code=400, detail="month must be 1-12")
    # build iso date as first of month
    iso = f"{year:04d}-{month:02d}-01"
    days = ml_service.predict_month_daily(crop, soil, location, iso)
    return [DayData(**d) for d in days]

# Calendar helpers
@router.get("/calendar/day", response_model=CalendarDayResponse)
def calendar_day(date: str = Query(...)):
    """
    date: ISO date "YYYY-MM-DD"
    """
    try:
        dt = datetime.fromisoformat(date)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format")
    info = suggested_day_info(dt.year, dt.month, dt.day)
    return CalendarDayResponse(**info)

@router.get("/calendar/month", response_model=List[CalendarDayResponse])
def calendar_month(year: int = Query(...), month: int = Query(...)):
    if not (1 <= month <= 12):
        raise HTTPException(status_code=400, detail="month must be 1-12")
    from calendar import monthrange
    days_in_month = monthrange(year, month)[1]
    out = [suggested_day_info(year, month, d) for d in range(1, days_in_month + 1)]
    return [CalendarDayResponse(**i) for i in out]
