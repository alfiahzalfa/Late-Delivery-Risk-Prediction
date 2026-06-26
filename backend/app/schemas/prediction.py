from pydantic import BaseModel, Field


class PredictS1Request(BaseModel):
    """Input untuk Skenario 1 — Pre-Shipment (info minimal saat order masuk)."""
    days_for_shipment_scheduled: int   = Field(..., ge=0, example=4)
    shipping_mode:               str   = Field(..., example="Standard Class")
    order_type:                  str   = Field(..., example="DEBIT")
    order_item_quantity:         int   = Field(..., ge=1, example=3)
    product_price:               float = Field(..., ge=0, example=199.99)
    order_item_discount_rate:    float = Field(..., ge=0, le=1, example=0.1)


class PredictS2Request(BaseModel):
    """Input untuk Skenario 2 — Full Order Profile (profil lengkap order)."""
    days_for_shipment_scheduled: int   = Field(..., ge=0, example=4)
    shipping_mode:               str   = Field(..., example="Standard Class")
    order_type:                  str   = Field(..., example="DEBIT")
    order_item_quantity:         int   = Field(..., ge=1, example=3)
    product_price:               float = Field(..., ge=0, example=199.99)
    order_item_discount_rate:    float = Field(..., ge=0, le=1, example=0.1)
    order_item_profit_ratio:     float = Field(..., example=0.25)
    order_region:                str   = Field(..., example="Western Europe")
    customer_state:              str   = Field(..., example="CA")


class PredictResponse(BaseModel):
    """Output hasil prediksi."""
    scenario:         str
    prediction:       int
    prediction_label: str
    probability:      float
    input_data:       dict
    timestamp:        str
