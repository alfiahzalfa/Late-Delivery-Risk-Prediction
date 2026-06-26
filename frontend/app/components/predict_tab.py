import streamlit as st
import requests
from frontend.app.utils.api_client import predict
from frontend.app.utils.charts import render_gauge_chart


def render_predict_tab(scenario: str):
    """Tampilkan tab prediksi — form input + hasil prediksi."""
    is_s2 = scenario == "s2"

    st.subheader("Input Data Order")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Informasi Pengiriman**")
        days_shipment = st.number_input(
            "Days for Shipment (Scheduled)", min_value=0, max_value=30, value=4,
            help="Estimasi hari pengiriman yang dijadwalkan",
        )
        shipping_mode = st.selectbox(
            "Shipping Mode",
            ["Standard Class", "Second Class", "First Class", "Same Day"],
        )
        order_type = st.selectbox(
            "Payment Type",
            ["DEBIT", "TRANSFER", "PAYMENT", "CASH"],
        )

    with col2:
        st.markdown("**Informasi Order**")
        order_qty = st.number_input(
            "Order Item Quantity", min_value=1, max_value=100, value=3
        )
        product_price = st.number_input(
            "Product Price ($)", min_value=0.0, max_value=2000.0, value=199.99, step=0.01
        )
        discount_rate = st.slider(
            "Order Item Discount Rate", min_value=0.0, max_value=1.0, value=0.1,
            step=0.01, format="%.2f",
        )

    # Field tambahan untuk Skenario 2
    profit_ratio  = None
    order_region  = None
    customer_state = None

    if is_s2:
        st.markdown("---")
        st.markdown("**Informasi Tambahan (Skenario 2)**")
        col3, col4 = st.columns(2)
        with col3:
            profit_ratio = st.number_input(
                "Order Item Profit Ratio", min_value=-1.0, max_value=1.0,
                value=0.25, step=0.01,
            )
            order_region = st.selectbox("Order Region", [
                "Western Europe", "Central America", "Oceania", "Eastern Asia",
                "South America", "Eastern Europe", "West Asia", "West Africa",
                "South Asia", "Southern Europe", "North America", "Central Africa",
                "Northern Europe", "East Africa", "Southeast Asia", "Caribbean",
                "South of  the  America", "North Africa",
            ])
        with col4:
            customer_state = st.text_input(
                "Customer State", value="CA",
                help="Kode negara bagian pelanggan, contoh: CA, NY, TX",
            )

    st.markdown("---")
    predict_btn = st.button("🔍 Prediksi Risiko", type="primary", use_container_width=True)

    if predict_btn:
        payload = {
            "days_for_shipment_scheduled": days_shipment,
            "shipping_mode"              : shipping_mode,
            "order_type"                 : order_type,
            "order_item_quantity"        : order_qty,
            "product_price"              : product_price,
            "order_item_discount_rate"   : discount_rate,
        }
        if is_s2:
            payload.update({
                "order_item_profit_ratio": profit_ratio,
                "order_region"           : order_region,
                "customer_state"         : customer_state,
            })

        with st.spinner("Memproses prediksi..."):
            try:
                result = predict(scenario, payload)
                pred   = result["prediction"]
                proba  = result["probability"]
                label  = result["prediction_label"]

                st.markdown("### 📊 Hasil Prediksi")
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric("Prediksi", label)
                with r2:
                    st.metric("Probabilitas Terlambat", f"{proba*100:.1f}%")
                with r3:
                    st.metric("Skenario", result["scenario"].split(" - ")[0])

                # Risk card
                if pred == 1:
                    st.markdown(f"""
                    <div class="risk-high">
                    <h4>⚠️ Risiko Tinggi — Berpotensi Terlambat</h4>
                    <p>Probabilitas keterlambatan: <strong>{proba*100:.1f}%</strong></p>
                    <p>Pertimbangkan untuk mengubah mode pengiriman atau memprioritaskan order ini.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-low">
                    <h4>✅ Risiko Rendah — Pengiriman Tepat Waktu</h4>
                    <p>Probabilitas keterlambatan: <strong>{proba*100:.1f}%</strong></p>
                    <p>Order ini diprediksi akan sampai tepat waktu.</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("#### Probabilitas Risiko")
                render_gauge_chart(proba)

            except requests.exceptions.ConnectionError:
                st.error("❌ Tidak bisa terhubung ke API. Pastikan FastAPI sudah berjalan.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
