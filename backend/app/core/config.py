import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "supply_chain")

# Path ke model .pkl
BASE_DIR = Path(__file__).resolve().parents[3]
MODELS_DIR = BASE_DIR / "models"
MODEL_S1_PATH = MODELS_DIR / "baseline_best_s1.pkl"
MODEL_S2_PATH = MODELS_DIR / "baseline_best_s2.pkl"

# Kategori untuk OHE (CASH jadi baseline)
TYPE_CATEGORIES = ["DEBIT", "PAYMENT", "TRANSFER"]
REGION_CATEGORIES: list[str] = []
STATE_CATEGORIES: list[str] = []

# Label encoding Shipping Mode
SHIPPING_ORDER = {
    "Same Day": 0,
    "First Class": 1,
    "Second Class": 2,
    "Standard Class": 3,
}
