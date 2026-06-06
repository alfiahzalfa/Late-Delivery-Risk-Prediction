import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

# MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = os.getenv("MONGO_DB", "supply_chain")

client     = None
db         = None
collection = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, db, collection
    client     = AsyncIOMotorClient(MONGO_URI)
    db         = client[DB_NAME]
    collection = db["predictions"]
    print(f"✅ MongoDB connected: {DB_NAME}")
    yield
    client.close()
    print("MongoDB disconnected")

# Models
MODEL_S1 = joblib.load("models/baseline_best_s1.pkl")
MODEL_S2 = joblib.load("models/baseline_best_s2.pkl")

# Skenario 1
TYPE_CATEGORIES   = ['DEBIT', 'PAYMENT', 'TRANSFER']   
# Skenario 2
REGION_CATEGORIES = []  
STATE_CATEGORIES  = []

SHIPPING_ORDER = {
    'Same Day'      : 0,
    'First Class'   : 1,
    'Second Class'  : 2,
    'Standard Class': 3,
}

# App
app = FastAPI(
    title="Late Delivery Risk Prediction API",
    description="Prediksi risiko keterlambatan pengiriman — Skenario 1 & 2",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas
class PredictS1Request(BaseModel):
    days_for_shipment_scheduled : int   = Field(..., ge=0, example=4)
    shipping_mode               : str   = Field(..., example="Standard Class")
    order_type                  : str   = Field(..., example="DEBIT")
    order_item_quantity         : int   = Field(..., ge=1, example=3)
    product_price               : float = Field(..., ge=0, example=199.99)
    order_item_discount_rate    : float = Field(..., ge=0, le=1, example=0.1)

class PredictS2Request(BaseModel):
    days_for_shipment_scheduled : int   = Field(..., ge=0, example=4)
    shipping_mode               : str   = Field(..., example="Standard Class")
    order_type                  : str   = Field(..., example="DEBIT")
    order_item_quantity         : int   = Field(..., ge=1, example=3)
    product_price               : float = Field(..., ge=0, example=199.99)
    order_item_discount_rate    : float = Field(..., ge=0, le=1, example=0.1)
    order_item_profit_ratio     : float = Field(..., example=0.25)
    order_region                : str   = Field(..., example="Western Europe")
    customer_state              : str   = Field(..., example="CA")

class PredictResponse(BaseModel):
    scenario        : str
    prediction      : int
    prediction_label: str
    probability     : float
    input_data      : dict
    timestamp       : str

# Helpers
def _encode_s1(req: PredictS1Request) -> pd.DataFrame:
    row = {
        'Days for shipment (scheduled)': req.days_for_shipment_scheduled,
        'Shipping Mode'                : SHIPPING_ORDER.get(req.shipping_mode, 3),
        'Order Item Quantity'          : req.order_item_quantity,
        'Product Price'                : req.product_price,
        'Order Item Discount Rate'     : req.order_item_discount_rate,
    }
    # OHE Type (drop_first=True → CASH jadi baseline)
    for cat in TYPE_CATEGORIES:
        row[f'Type_{cat}'] = int(req.order_type == cat)

    return pd.DataFrame([row])

def _encode_s2(req: PredictS2Request) -> pd.DataFrame:
    discount_per_unit = req.order_item_discount_rate * req.product_price
    row = {
        'Days for shipment (scheduled)': req.days_for_shipment_scheduled,
        'Shipping Mode'                : SHIPPING_ORDER.get(req.shipping_mode, 3),
        'Order Item Quantity'          : req.order_item_quantity,
        'Order Item Discount Rate'     : req.order_item_discount_rate,
        'Product Price'                : req.product_price,
        'Order Item Profit Ratio'      : req.order_item_profit_ratio,
        'discount_per_unit'            : discount_per_unit,
    }
    # OHE Type
    for cat in TYPE_CATEGORIES:
        row[f'Type_{cat}'] = int(req.order_type == cat)
    # OHE Order Region
    for cat in REGION_CATEGORIES:
        row[f'Order Region_{cat}'] = int(req.order_region == cat)
    # OHE Customer State
    for cat in STATE_CATEGORIES:
        row[f'Customer State_{cat}'] = int(req.customer_state == cat)

    return pd.DataFrame([row])

async def _save_to_mongo(scenario: str, input_data: dict, prediction: int,
                          probability: float):
    if collection is None:
        return
    doc = {
        "scenario"   : scenario,
        "input"      : input_data,
        "prediction" : prediction,
        "probability": round(probability, 4),
        "label"      : "Terlambat" if prediction == 1 else "Tidak Terlambat",
        "timestamp"  : datetime.utcnow(),
    }
    await collection.insert_one(doc)

# Endpoints
@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "Late Delivery Risk Prediction API"}

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.post("/predict/s1", response_model=PredictResponse, tags=["Prediction"])
async def predict_s1(req: PredictS1Request):
    try:
        X = _encode_s1(req)
        # align kolom ke model
        model_cols = MODEL_S1.feature_names_in_ if hasattr(MODEL_S1, 'feature_names_in_') else X.columns
        X = X.reindex(columns=model_cols, fill_value=0)

        pred  = int(MODEL_S1.predict(X)[0])
        proba = float(MODEL_S1.predict_proba(X)[0][1])
        label = "Terlambat" if pred == 1 else "Tidak Terlambat"
        ts    = datetime.utcnow().isoformat()

        await _save_to_mongo("S1", req.model_dump(), pred, proba)

        return PredictResponse(
            scenario=        "Skenario 1 - Pre-Shipment",
            prediction=      pred,
            prediction_label=label,
            probability=     round(proba, 4),
            input_data=      req.model_dump(),
            timestamp=       ts,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/s2", response_model=PredictResponse, tags=["Prediction"])
async def predict_s2(req: PredictS2Request):
    try:
        X = _encode_s2(req)
        model_cols = MODEL_S2.feature_names_in_ if hasattr(MODEL_S2, 'feature_names_in_') else X.columns
        X = X.reindex(columns=model_cols, fill_value=0)

        pred  = int(MODEL_S2.predict(X)[0])
        proba = float(MODEL_S2.predict_proba(X)[0][1])
        label = "Terlambat" if pred == 1 else "Tidak Terlambat"
        ts    = datetime.utcnow().isoformat()

        await _save_to_mongo("S2", req.model_dump(), pred, proba)

        return PredictResponse(
            scenario=        "Skenario 2 - Full Order Profile",
            prediction=      pred,
            prediction_label=label,
            probability=     round(proba, 4),
            input_data=      req.model_dump(),
            timestamp=       ts,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history", tags=["Database"])
async def get_history(limit: int = 20):
    """Ambil riwayat prediksi dari MongoDB."""
    if collection is None:
        raise HTTPException(status_code=503, detail="Database tidak tersedia")
    cursor = collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
    docs   = await cursor.to_list(length=limit)
    return {"count": len(docs), "data": docs}