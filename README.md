# 🏡 Prediksi Harga Rumah di Jabodetabek dengan Machine Learning

Repositori ini berisi empat eksperimen prediksi harga rumah menggunakan model **Random Forest** dan **Gradient Boosting** berdasarkan dataset properti wilayah Jabodetabek. Dataset ini berasal dari [Kaggle](https://www.kaggle.com) dengan judul *House Prices in Greater Jakarta Area*. Tujuannya adalah mengevaluasi performa berbagai pendekatan pemodelan dan preprocessing terhadap akurasi prediksi harga rumah.

---

## 📁 Dataset

- **Nama file**: `jabodetabek_house_price.csv`
- **Target**: `price_in_rp`
- **Fitur**: Luas bangunan, luas tanah, jumlah kamar tidur & mandi, tahun dibangun, lokasi (lat, long), fasilitas, dan lainnya.

---

## ⚙️ Eksperimen

### ✅ Eksperimen 1: Random Forest tanpa Preprocessing

- **Deskripsi**: Menggunakan fitur numerik saja tanpa pembersihan data, tanpa encoding, dan tanpa scaling.
- **Model**: `RandomForestRegressor`
- **Hasil**:
  - **MAE**: 1,589,833,710
  - **RMSE**: 6,487,119,261
  - **R² Score**: 0.4230
- **Kesimpulan**: Akurasi rendah karena banyak fitur diabaikan dan data belum dibersihkan.

---

### ✅ Eksperimen 2: Random Forest dengan Preprocessing Lengkap

- **Deskripsi**: Data dibersihkan, missing value ditangani, encoding kategori dilakukan, fitur baru dibuat, dan model dituning.
- **Model**: `RandomForestRegressor + GridSearchCV`
- **Best Params**:

```python
{
  'regressor__max_depth': None,
  'regressor__max_features': 'sqrt',
  'regressor__min_samples_leaf': 1,
  'regressor__min_samples_split': 2,
  'regressor__n_estimators': 200
}
