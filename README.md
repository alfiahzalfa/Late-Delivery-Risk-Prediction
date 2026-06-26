# Late Delivery Risk Prediction

Proyek ini dibuat untuk memprediksi risiko keterlambatan pengiriman di supply chain menggunakan machine learning.

Dataset: https://data.mendeley.com/datasets/8gx2fvg2k6/3

---

## Dua Skenario Prediksi

Membuat dua skenario karena informasi yang tersedia saat order masuk dan setelah diproses berbeda, maka:

- **Skenario 1 (Pre-Shipment)** — menggunakan informasi yang minimalis yaitu estimasi hari pengiriman, metode pengiriman, tipe pembayaran, kuantitas order, harga produk, dan discount rate.

- **Skenario 2 (Full Order Profile)** — menambahkan beberapa informasi, seperti rasio keuntungan per item, diskon per unit, wilayah melakukan order, dan customer state.

---

## Algoritma yang Digunakan

- **Backend**: FastAPI 
- **Frontend**: Streamlit
- **Model**: Logistic Regression, Decision Tree, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost 
- **Hyperparameter Optimization**: Optuna, RandomizedSearchCV, HalvingRandomizedSearchCV 
- **Evaluasi Metrik**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC

---

## Struktur Folder

```
├── backend/        # FastAPI — API prediksi dan riwayat
├── frontend/       # Streamlit — UI input dan visualisasi
├── notebooks/      # Proses eksplorasi dan training
├── src/            # Modul Python yang dipakai di notebook
└── tests/          # Unit test API
```

---

## Cara Jalankan Lokal

Pastikan sudah punya Python 3.10+ dan MongoDB

**1. Clone dan install dependencies**
```bash
git clone https://github.com/alfiahzalfa/Late-Delivery-Risk-Prediction.git
cd Late-Delivery-Risk-Prediction
pip install -r requirements.txt
```

**2. Siapkan file `.env`**
```
MONGO_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/<dbname>?appName=<appname>
MONGO_DB=supply_chain
API_URL=http://localhost:8000
```

**3. Taruh model `.pkl` di `backend/models/`**

File model tidak disertakan di repo karena ukurannya besar. Download dari [Google Drive](https://drive.google.com/drive/folders/1HtvenhO5p8Raip67o7ISsKmIhodXpKfb?usp=sharing) dan letakkan di folder `backend/models/`.

**4. Jalankan backend**
```bash
uvicorn backend.app.main:app --reload
```

**5. Jalankan frontend**
```bash
streamlit run frontend/app/streamlit_app.py
```

Buka `http://localhost:8501` di browser.

