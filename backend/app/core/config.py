import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── MongoDB ────────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = os.getenv("MONGO_DB", "supply_chain")

# ── Model paths ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parents[3]   # backend/
MODELS_DIR  = BASE_DIR / "models"
MODEL_S1_PATH = MODELS_DIR / "baseline_best_s1.pkl"
MODEL_S2_PATH = MODELS_DIR / "baseline_best_s2.pkl"

# ── Encoding constants ─────────────────────────────────────────────────────────
# OHE Type (drop_first=True → CASH jadi baseline)
TYPE_CATEGORIES = ["DEBIT", "PAYMENT", "TRANSFER"]

# OHE Order Region & Customer State (diisi dari kategori training)
REGION_CATEGORIES: list[str] = []
STATE_CATEGORIES:  list[str] = []

# Label-encoding Shipping Mode
SHIPPING_ORDER: dict[str, int] = {
    "Same Day"      : 0,
    "First Class"   : 1,
    "Second Class"  : 2,
    "Standard Class": 3,
}
