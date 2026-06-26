from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.database import lifespan
from backend.app.routes import predict, history

# ── App ────────────────────────────────────────────────────────────────────────
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

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(predict.router, prefix="/predict")
app.include_router(history.router)


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "Late Delivery Risk Prediction API"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
