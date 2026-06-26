from fastapi import APIRouter, HTTPException
from backend.app.core.database import get_collection

router = APIRouter()


@router.get("/history", tags=["Database"])
async def get_history(limit: int = 20):
    """Ambil riwayat prediksi dari MongoDB."""
    col = get_collection()
    if col is None:
        raise HTTPException(status_code=503, detail="Database tidak tersedia")
    cursor = col.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
    docs   = await cursor.to_list(length=limit)
    return {"count": len(docs), "data": docs}
