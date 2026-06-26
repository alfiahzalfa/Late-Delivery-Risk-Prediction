import pandas as pd
import streamlit as st
from requests.exceptions import ConnectionError as RequestsConnectionError
from frontend.app.utils.api_client import get_history
from frontend.app.utils.charts import render_distribution_chart


def render_history_tab():
    st.subheader("📋 Riwayat Prediksi")

    # init session state untuk refresh counter
    if "history_refresh" not in st.session_state:
        st.session_state["history_refresh"] = 0

    col1, col2 = st.columns([1, 3])
    with col1:
        limit = st.number_input("Tampilkan N terakhir", min_value=5, max_value=100, value=20)
    with col2:
        if st.button("🔄 Refresh", width="content"):
            st.session_state["history_refresh"] += 1
            st.rerun()

    try:
        data = get_history(limit)
        if not data:
            st.info("Belum ada riwayat prediksi.")
            return

        rows = [
            {
                "Timestamp": d.get("timestamp", "")[:19].replace("T", " "),
                "Skenario": d.get("scenario", ""),
                "Prediksi": d.get("label", ""),
                "Probabilitas": f'{d.get("probability", 0)*100:.1f}%',
            }
            for d in data
        ]
        df_hist = pd.DataFrame(rows)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        st.markdown("#### Distribusi Prediksi")
        render_distribution_chart(df_hist)

    except RequestsConnectionError:
        st.error("❌ Tidak bisa terhubung ke API.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
