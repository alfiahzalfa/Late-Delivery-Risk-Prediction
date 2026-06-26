import joblib
import pandas as pd
from datetime import datetime

from backend.app.core.config import (
    MODEL_S1_PATH, MODEL_S2_PATH,
    TYPE_CATEGORIES, REGION_CATEGORIES, STATE_CATEGORIES,
    SHIPPING_ORDER,
)
from backend.app.core.database import get_collection
from backend.app.schemas.prediction import PredictS1Request, PredictS2Request

# ── Load model saat modul diimport ────────────────────────────────────────────
MODEL_S1 = joblib.load(MODEL_S1_PATH)
MODEL_S2 = joblib.load(MODEL_S2_PATH)


# ── Encoding ──────────────────────────────────────────────────────────────────
def encode_s1(req: PredictS1Request) -> pd.DataFrame:
    """Encode input Skenario 1 menjadi DataFrame siap prediksi."""
    row = {
        "Days for shipment (scheduled)": req.days_for_shipment_scheduled,
        "Shipping Mode"                : SHIPPING_ORDER.get(req.shipping_mode, 3),
        "Order Item Quantity"          : req.order_item_quantity,
        "Product Price"                : req.product_price,
        "Order Item Discount Rate"     : req.order_item_discount_rate,
    }
    # OHE Type (drop_first=True → CASH jadi baseline)
    for cat in TYPE_CATEGORIES:
        row[f"Type_{cat}"] = int(req.order_type == cat)

    return pd.DataFrame([row])


def encode_s2(req: PredictS2Request) -> pd.DataFrame:
    """Encode input Skenario 2 menjadi DataFrame siap prediksi."""
    discount_per_unit = req.order_item_discount_rate * req.product_price
    row = {
        "Days for shipment (scheduled)": req.days_for_shipment_scheduled,
        "Shipping Mode"                : SHIPPING_ORDER.get(req.shipping_mode, 3),
        "Order Item Quantity"          : req.order_item_quantity,
        "Order Item Discount Rate"     : req.order_item_discount_rate,
        "Product Price"                : req.product_price,
        "Order Item Profit Ratio"      : req.order_item_profit_ratio,
        "discount_per_unit"            : discount_per_unit,
    }
    # OHE Type
    for cat in TYPE_CATEGORIES:
        row[f"Type_{cat}"] = int(req.order_type == cat)
    # OHE Order Region
    for cat in REGION_CATEGORIES:
        row[f"Order Region_{cat}"] = int(req.order_region == cat)
    # OHE Customer State
    for cat in STATE_CATEGORIES:
        row[f"Customer State_{cat}"] = int(req.customer_state == cat)

    return pd.DataFrame([row])


# ── Prediksi ──────────────────────────────────────────────────────────────────
def run_prediction(model, X: pd.DataFrame) -> tuple[int, float, str]:
    """Jalankan prediksi dan kembalikan (pred, proba, label)."""
    model_cols = (
        model.feature_names_in_ if hasattr(model, "feature_names_in_") else X.columns
    )
    X = X.reindex(columns=model_cols, fill_value=0)
    pred  = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    label = "Terlambat" if pred == 1 else "Tidak Terlambat"
    return pred, proba, label


# ── Simpan ke MongoDB ─────────────────────────────────────────────────────────
async def save_to_mongo(scenario: str, input_data: dict, prediction: int, probability: float):
    """Simpan hasil prediksi ke MongoDB."""
    col = get_collection()
    if col is None:
        return
    doc = {
        "scenario"   : scenario,
        "input"      : input_data,
        "prediction" : prediction,
        "probability": round(probability, 4),
        "label"      : "Terlambat" if prediction == 1 else "Tidak Terlambat",
        "timestamp"  : datetime.utcnow(),
    }
    await col.insert_one(doc)
