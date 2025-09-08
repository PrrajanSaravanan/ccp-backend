# app/services/data_service.py
from typing import Optional
from math import sin, pi
from datetime import date

CROPS = ["Maize", "Wheat", "Rice", "Soybean", "Cotton", "Sorghum"]

def pick_crop(seed: int) -> str:
    return CROPS[seed % len(CROPS)]

def required_water_pct_for_day(d: int) -> float:
    wave = round(50 + 40 * sin(((d + 4) / 31) * pi))
    return float(max(10, min(100, wave)))

def suggested_day_info(year: int, month: int, day: int):
    # Returns suggestedCrop and requiredWaterPct for this day (mirrors frontend heuristics)
    seed = day + month + year
    return {
        "date": f"{year:04d}-{month:02d}-{day:02d}",
        "suggestedCrop": pick_crop(seed),
        "requiredWaterPct": required_water_pct_for_day(day)
    }
