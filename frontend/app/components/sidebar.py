import streamlit as st
from frontend.app.utils.api_client import check_api_health


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🚚 Supply Chain\nRisk Prediction")
        st.markdown("---")

        scenario_label = st.radio(
            "Pilih Skenario",
            ["Skenario 1 — Pre-Shipment", "Skenario 2 — Full Order Profile"],
            help="S1: info minimal saat order masuk | S2: profil lengkap order",
        )

        st.markdown("---")
        st.markdown("**API Status**")
        if check_api_health():
            st.success("🟢 API Online")
        else:
            st.error("🔴 API Offline")

    return "s2" if "2" in scenario_label else "s1"

