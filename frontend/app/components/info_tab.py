import streamlit as st


def render_info_tab():
    """Tampilkan tab informasi model dan fitur yang digunakan."""
    st.subheader("ℹ️ Informasi Model")

    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.markdown("""
        **Skenario 1 — Pre-Shipment**
        - Digunakan saat order baru masuk
        - Fitur minimal yang tersedia sebelum pengiriman
        - Cocok untuk *early warning system*

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
        - Discount per Unit *(engineered)*
        - Order Region
        - Customer State
        """)
