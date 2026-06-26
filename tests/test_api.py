import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch

from backend.app.main import app


@pytest.mark.asyncio
async def test_health():
    with patch("backend.app.core.database.collection", None):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_predict_s1():
    with patch("backend.app.core.database.collection", None):
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