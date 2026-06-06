import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from datetime import datetime

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Config
st.set_page_config(
    page_title="Late Delivery Risk Prediction",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style
st.markdown("""
<style>
.risk-low p, .risk-low h4 { color: #085041 !important; }
.risk-high p, .risk-high h4 { color: #712B13 !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar 
    st.image("https://img.icons8.com/fluency/96/delivery.png", width=64)
    st.title("Supply Chain\nRisk Prediction")
    st.markdown("---")
    scenario = st.radio(
        "Pilih Skenario",
        ["Skenario 1 — Pre-Shipment", "Skenario 2 — Full Order Profile"],
        help="S1: info minimal saat order masuk | S2: profil lengkap order",
    )
    st.markdown("---")
    st.markdown("**API Status**")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.success("🟢 API Online")
        else:
            st.error("🔴 API Error")
    except Exception:
        st.error("🔴 API Offline")

# Main
st.title("🚚 Late Delivery Risk Prediction")
st.caption("Prediksi risiko keterlambatan pengiriman berbasis Machine Learning")
st.markdown("---")

tab_predict, tab_history, tab_info = st.tabs(["🎯 Prediksi", "📋 Riwayat", "ℹ️ Info Model"])

# Tab Prediksi
with tab_predict:

    is_s2 = "2" in scenario
    endpoint = "/predict/s2" if is_s2 else "/predict/s1"

    st.subheader("Input Data Order")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Informasi Pengiriman**")
        days_shipment = st.number_input(
            "Days for Shipment (Scheduled)", min_value=0, max_value=30, value=4,
            help="Estimasi hari pengiriman yang dijadwalkan"
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
            "Order Item Discount Rate", min_value=0.0, max_value=1.0, value=0.1, step=0.01,
            format="%.2f"
        )

    # Skenario 2
    if is_s2:
        st.markdown("---")
        st.markdown("**Informasi Tambahan (Skenario 2)**")
        col3, col4 = st.columns(2)
        with col3:
            profit_ratio = st.number_input(
                "Order Item Profit Ratio", min_value=-1.0, max_value=1.0, value=0.25, step=0.01
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
                help="Kode negara bagian pelanggan, contoh: CA, NY, TX"
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
                resp = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=10)
                resp.raise_for_status()
                result = resp.json()

                pred  = result["prediction"]
                proba = result["probability"]
                label = result["prediction_label"]

                # Hasil
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

                # Gauge chart
                st.markdown("#### Probabilitas Risiko")
                fig, ax = plt.subplots(figsize=(6, 1.2))
                fig.patch.set_facecolor('#FAFAFA')
                ax.barh(0, 1, color='#E8F5E9', height=0.5)
                ax.barh(0, proba, color='#E05A3A' if proba > 0.5 else '#2EAF7D', height=0.5)
                ax.axvline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6)
                ax.set_xlim(0, 1)
                ax.set_yticks([])
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                ax.spines[['top', 'right', 'left']].set_visible(False)
                ax.set_facecolor('#FAFAFA')
                ax.text(proba, 0, f' {proba*100:.1f}%', va='center', fontweight='bold',
                        color='#E05A3A' if proba > 0.5 else '#2EAF7D')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            except requests.exceptions.ConnectionError:
                st.error("❌ Tidak bisa terhubung ke API. Pastikan FastAPI sudah berjalan.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

#  Tab Riwayat
with tab_history:
    st.subheader("📋 Riwayat Prediksi")

    col_h1, col_h2 = st.columns([1, 3])
    with col_h1:
        limit = st.number_input("Tampilkan N terakhir", min_value=5, max_value=100, value=20)
    with col_h2:
        refresh = st.button("🔄 Refresh", use_container_width=False)

    if refresh or True:
        try:
            r = requests.get(f"{API_URL}/history?limit={limit}", timeout=5)
            r.raise_for_status()
            data = r.json().get("data", [])

            if not data:
                st.info("Belum ada riwayat prediksi.")
            else:
                rows = []
                for d in data:
                    rows.append({
                        "Timestamp" : d.get("timestamp", "")[:19].replace("T", " "),
                        "Skenario"  : d.get("scenario", ""),
                        "Prediksi"  : d.get("label", ""),
                        "Probabilitas": f'{d.get("probability", 0)*100:.1f}%',
                    })
                df_hist = pd.DataFrame(rows)
                st.dataframe(df_hist, use_container_width=True, hide_index=True)

                # Mini chart distribusi
                st.markdown("#### Distribusi Prediksi")
                counts = df_hist["Prediksi"].value_counts()
                fig2, ax2 = plt.subplots(figsize=(4, 3))
                colors = ['#E05A3A' if l == 'Terlambat' else '#2EAF7D' for l in counts.index]
                ax2.bar(counts.index, counts.values, color=colors, edgecolor='white')
                ax2.set_ylabel("Jumlah")
                ax2.spines[['top', 'right']].set_visible(False)
                ax2.set_facecolor('#FAFAFA')
                fig2.patch.set_facecolor('#FAFAFA')
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

        except requests.exceptions.ConnectionError:
            st.error("❌ Tidak bisa terhubung ke API.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Tab Info
with tab_info:
    st.subheader("ℹ️ Informasi Model")

    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.markdown("""
        **Skenario 1 — Pre-Shipment**
        - Digunakan saat order baru masuk
        - Fitur minimal yang tersedia sebelum pengiriman
        - Cocok untuk early warning system

        **Fitur yang digunakan:**
        - Days for Shipment (Scheduled)
        - Shipping Mode
        - Payment Type
        - Order Item Quantity
        - Product Price
        - Order Item Discount Rate
        """)

    with col_i2:
        st.markdown("""
        **Skenario 2 — Full Order Profile**
        - Digunakan setelah profil order lengkap tersedia
        - Akurasi lebih tinggi dengan lebih banyak fitur
        - Cocok untuk validasi sebelum dispatch

        **Fitur tambahan:**
        - Order Item Profit Ratio
        - Discount per Unit (engineered)
        - Order Region
        - Customer State
        """)
