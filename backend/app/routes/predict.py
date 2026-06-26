from datetime import datetime
from fastapi import APIRouter, HTTPException

from backend.app.schemas.prediction import PredictS1Request, PredictS2Request, PredictResponse
from backend.app.services.predictor import MODEL_S1, MODEL_S2, encode_s1, encode_s2, run_prediction, save_to_mongo

router = APIRouter()


@router.post("/s1", response_model=PredictResponse, tags=["Prediction"])
async def predict_s1(req: PredictS1Request):
    try:
        X = encode_s1(req)
        pred, proba, label = run_prediction(MODEL_S1, X)
        await save_to_mongo("S1", req.model_dump(), pred, proba)
        return PredictResponse(
            scenario="Skenario 1 - Pre-Shipment",
            prediction=pred,
            prediction_label=label,
            probability=round(proba, 4),
            input_data=req.model_dump(),
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/s2", response_model=PredictResponse, tags=["Prediction"])
async def predict_s2(req: PredictS2Request):
    try:
        X = encode_s2(req)
        pred, proba, label = run_prediction(MODEL_S2, X)
        await save_to_mongo("S2", req.model_dump(), pred, proba)
        return PredictResponse(
            scenario="Skenario 2 - Full Order Profile",
            prediction=pred,
            prediction_label=label,
            probability=round(proba, 4),
            input_data=req.model_dump(),
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
