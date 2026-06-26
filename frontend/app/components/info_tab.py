import streamlit as st


def render_info_tab():
    st.subheader("ℹ️ Informasi Model")

    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.markdown("### 📦 Skenario 1 — Pre-Shipment")
        st.info("Digunakan saat order **baru masuk**, sebelum pengiriman dimulai.")

        st.markdown("**🤖 Algoritma:** `Logistic Regression`")
        st.markdown("""
        **Skenario 1 — Pre-Shipment**
        - Digunakan saat order baru masuk
        - Fitur minimal yang tersedia sebelum pengiriman
        - Cocok untuk *early warning system*

        **Fitur yang digunakan:**
        | Fitur | Keterangan |
        |-------|-----------|
        | Days for Shipment (Scheduled) | Estimasi hari kirim |
        | Shipping Mode | Metode pengiriman |
        | Payment Type | Tipe pembayaran |
        | Order Item Quantity | Jumlah item |
        | Product Price | Harga produk |
        | Order Item Discount Rate | Diskon |
        """)

    with col_i2:
        st.markdown("### 🗂️ Skenario 2 — Full Order Profile")
        st.info("Digunakan setelah profil order **lengkap tersedia**, sebelum dispatch.")

        st.markdown("**🤖 Algoritma:** `Extra Trees Classifier`")
        st.markdown("""
        **Kapan digunakan:**
        - Validasi akhir sebelum pengiriman
        - Ada penambahan beberapa fitur yang lebih lengkap

        **Fitur tambahan (selain S1):**
        | Fitur | Keterangan |
        |-------|-----------|
        | Order Item Profit Ratio | Rasio keuntungan per item |
        | Discount per Unit | Fitur turunan (discount × price) |
        | Order Region | Wilayah order |
        | Customer State | Negara bagian pelanggan |
        """)
