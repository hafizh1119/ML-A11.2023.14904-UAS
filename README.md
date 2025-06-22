
# 🏡 Prediksi Harga Rumah di Jabodetabek dengan Machine Learning

Repositori ini berisi empat eksperimen prediksi harga rumah menggunakan model **Random Forest** dan **Gradient Boosting** berdasarkan dataset properti wilayah Jabodetabek. Dataset ini berasal dari [Kaggle](https://www.kaggle.com) dengan judul *House Prices in Greater Jakarta Area*. Tujuannya adalah mengevaluasi performa berbagai pendekatan pemodelan dan preprocessing terhadap akurasi prediksi harga rumah.

---

## 📁 Dataset

- **Nama file**: `jabodetabek_house_price.csv`
- **Target**: `price_in_rp`
- **Fitur**: 'district', 'city', 'lat', 'long', 'property_type', 'bedrooms', 'bathrooms','land_size_m2', 'building_size_m2', 'carports', 'certificate', 'electricity','maid_bedrooms', 'maid_bathrooms', 'floors', 'building_age', 'property_condition','building_orientation', 'garages', 'furnishing', 'has_pool', 'has_security', 'has_park'


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
```

- **Hasil**:
  - **MAE**: 0.1849
  - **RMSE**: 0.3170
  - **R² Score**: 0.9192
- **Kesimpulan**: Preprocessing menyeluruh dan tuning parameter mampu meningkatkan akurasi secara signifikan.

---

### ✅ Eksperimen 3: Gradient Boosting dengan Preprocessing Lengkap

- **Deskripsi**: Mengganti model dengan `GradientBoostingRegressor` dan melakukan tuning.
- **Best Params**:

```python
{
  'regressor__learning_rate': 0.05,
  'regressor__max_depth': 5,
  'regressor__n_estimators': 200
}
```

- **Hasil**:
  - **MAE**: 0.2004
  - **RMSE**: 0.3127
  - **R² Score**: 0.9214
- **Kesimpulan**: Gradient Boosting memberi hasil akurasi tertinggi dan sangat cocok untuk data yang kompleks dan non-linear.

---

### ✅ Eksperimen 4: Random Forest dengan Seleksi Top 20 Fitur

- **Deskripsi**: Mengambil 20 fitur terbaik dari model Random Forest dan melatih ulang model hanya dengan fitur tersebut.
- **Model**: `RandomForestRegressor` (tanpa tuning ulang)
- **Hasil**:
  - **MAE**: 0.2038
  - **RMSE**: 0.3512
  - **R² Score**: 0.9009
- **Kesimpulan**: Mengurangi jumlah fitur menyederhanakan model, tetapi sedikit menurunkan akurasi. Masih cocok digunakan jika performa dan interpretasi lebih diutamakan.

---

## 📊 Perbandingan Hasil

| Eksperimen | Model                 | MAE     | RMSE    | R² Score |
|------------|------------------------|---------|---------|----------|
| 1          | Random Forest (raw)    | 1.59B   | 6.48B   | 0.4230   |
| 2          | Random Forest (full)   | 0.1849  | 0.3170  | 0.9192   |
| 3          | Gradient Boosting      | 0.2004  | 0.3127  | 0.9214   |
| 4          | Random Forest (Top 20) | 0.2038  | 0.3512  | 0.9009   |

---

## 📌 Kesimpulan Umum

Eksperimen membuktikan bahwa **kualitas preprocessing dan pemilihan fitur sangat mempengaruhi akurasi model**. Model Gradient Boosting sedikit lebih unggul dari Random Forest dalam hal akurasi, namun Random Forest dengan fitur penuh tetap memberikan hasil yang sangat kompetitif. Penggunaan seleksi fitur dapat dijadikan alternatif untuk efisiensi jika diperlukan.

---

## 📦 Tools & Library

- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn

---

## 📝 Author

Eksperimen ini dilakukan sebagai bagian dari studi analisis prediktif harga rumah wilayah Jabodetabek menggunakan pendekatan *supervised learning regression*.
