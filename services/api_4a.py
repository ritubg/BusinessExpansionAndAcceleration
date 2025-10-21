from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
import os
import pickle

app = FastAPI()

RESULTS_PATH = "D:/SEMESTER 5/ML Lab/BusinessAcceralation/models/artifacts/4a_results.pkl"
REVENUE_PLOT ="D:/SEMESTER 5/ML Lab/BusinessAcceralation/models/artifacts/4a_revenue_forecast.png"
LIQUIDITY_PLOT="D:/SEMESTER 5/ML Lab/BusinessAcceralation/models/artifacts/4a_liquidity_forecast.png"

def safe_json(d):
    import numpy as np
    import pandas as pd
    import datetime

    if isinstance(d, (np.ndarray, pd.Series, list)):
        return d.tolist() if hasattr(d, "tolist") else list(d)
    elif isinstance(d, (np.integer, int)):
        return int(d)
    elif isinstance(d, (np.floating, float)):
        return float(d)
    elif isinstance(d, (np.datetime64, datetime.datetime)):
        return str(d)
    elif isinstance(d, dict):
        return {k: safe_json(v) for k, v in d.items()}
    else:
        return str(d)

@app.get("/")
def root():
    return {"message": "API 4A (Revenue & Liquidity Forecast) is running. Use /metrics, /historical, /forecast, /plots endpoints."}


@app.get("/metrics")
def metrics():
    if not os.path.exists(RESULTS_PATH):
        return {"status": "error", "error": "Results not found. Run model_4a.py first."}
    try:
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        return JSONResponse(content={
            "status": "success",
            "best_model_name": results.get("best_model"),
            "metrics": results.get("metrics", {})
        })
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/historical")
def historical():
    if not os.path.exists(RESULTS_PATH):
        return {"status": "error", "error": "Results not found. Run model_4a.py first."}
    try:
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        return JSONResponse(content=safe_json(results.get("historical_revenue", [])))
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/forecast")
def forecast():
    if not os.path.exists(RESULTS_PATH):
        return {"status": "error", "error": "Results not found. Run model_4a.py first."}
    try:
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        return JSONResponse(content={
            "future_dates": results.get("future_dates", []),
            "future_pred": results.get("future_pred", []),
            "liquidity_forecast": results.get("liquidity_forecast", [])
        })
    except Exception as e:
        return {"status": "error", "error": str(e)}



@app.get("/plots/revenue")
def plot_revenue():
    if not os.path.exists(REVENUE_PLOT):
        return {"status": "error", "error": "Revenue plot not found. Run model_4a.py first."}
    return FileResponse(REVENUE_PLOT, media_type="image/png")

@app.get("/plots/liquidity")
def plot_liquidity():
    if not os.path.exists(LIQUIDITY_PLOT):
        return {"status": "error", "error": "Liquidity plot not found. Run model_4a.py first."}
    return FileResponse(LIQUIDITY_PLOT, media_type="image/png")
