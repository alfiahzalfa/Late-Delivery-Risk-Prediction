import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

@pytest.mark.asyncio
async def test_health():
    with patch("app.main.collection", None):
        from app.main import app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

@pytest.mark.asyncio
async def test_predict_s1():
    with patch("app.main.collection", None):
        from app.main import app
        payload = {
            "days_for_shipment_scheduled": 4,
            "shipping_mode"              : "Standard Class",
            "order_type"                 : "DEBIT",
            "order_item_quantity"        : 3,
            "product_price"              : 199.99,
            "order_item_discount_rate"   : 0.1,
        }
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post("/predict/s1", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1