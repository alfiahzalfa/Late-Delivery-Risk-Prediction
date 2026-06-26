import streamlit as st
from frontend.app.components.sidebar import render_sidebar
from frontend.app.components.predict_tab import render_predict_tab
from frontend.app.components.history_tab import render_history_tab
from frontend.app.components.info_tab import render_info_tab

# config
st.set_page_config(
    page_title="Late Delivery Risk Prediction",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# style
st.markdown("""
<style>
.risk-low p, .risk-low h4  { color: #085041 !important; }
.risk-high p, .risk-high h4 { color: #712B13 !important; }
</style>
""", unsafe_allow_html=True)

# sidebar
scenario = render_sidebar()

# main
st.title("🚚 Late Delivery Risk Prediction")
st.caption("Prediksi risiko keterlambatan pengiriman berbasis Machine Learning")
st.markdown("---")

tab_predict, tab_history, tab_info = st.tabs(["🎯 Prediksi", "📋 Riwayat", "ℹ️ Info Model"])

with tab_predict:
    render_predict_tab(scenario)

with tab_history:
    render_history_tab()

with tab_info:
    render_info_tab()
