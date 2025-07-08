
# 🏡 PREDIKSI HARGA RUMAH DI JABODETABEK MENGGUNAKAN RANDOM FOREST & XGBOOST

Pertumbuhan sektor properti di wilayah Jabodetabek semakin pesat setiap tahunnya. Namun, harga rumah di kawasan ini sangat bervariasi tergantung berbagai faktor seperti lokasi, luas tanah, jumlah kamar, kondisi bangunan, hingga status sertifikat. Ketidakteraturan harga ini menyulitkan calon pembeli maupun penjual untuk menentukan harga wajar.

Untuk menjawab permasalahan ini, proyek ini membandingkan dua model machine learning populer, Random Forest dan XGBoost, dalam melakukan prediksi harga rumah berbasis data historis properti.

---
# 🎯 TUJUAN PROYEK
  1. Mengolah dan membersihkan data harga rumah dari Jabodetabek.

  2. Melakukan eksplorasi data (EDA) dan rekayasa fitur (feature engineering).

  3. Membangun dan membandingkan performa dua model regresi:

      - Random Forest Regressor

      - XGBoost Regressor

  4. Mengevaluasi performa model menggunakan metrik:

      - MAE (Mean Absolute Error)

      - RMSE (Root Mean Squared Error)

      - R² Score
# 🗂️ DATASET

Dataset: jabodetabek_house_price.csv

Berisi lebih dari 10.000 entri properti dengan atribut:

✳️ Fitur Numerik:
lat, long, bedrooms, bathrooms, land_size_m2, building_size_m2, carports, maid_bedrooms, maid_bathrooms,
floors, building_age, year_built, garages, building_density, room_ratio, bed_bath_ratio.

🔤 Fitur Kategorikal:
district, city, property_type, certificate, furnishing, propety_condition, building_orientation

➕ Fitur Turunan:
building_density, room_ratio, bed_bath_ratio

📌 Gambar: Distribusi Harga Rumah
![Screenshot 2025-07-08 221956](https://github.com/user-attachments/assets/342ccdb8-054e-40c0-bb87-18bfdbe54750)

📌 Gambar: Heatmap Korelasi Fitur Numerik
![image](https://github.com/user-attachments/assets/b071e701-c67d-42cf-b8c1-e5474e0e7080)

# 🔧 PROSES PENYELESAIAN

📌 Gambar: Diagram Alur Model

Langkah-langkah utama:

  1. Import data dan eksplorasi awal.

  2. Bersihkan data dari outlier & missing value.

  3. Feature engineering dan preprocessing (standarisasi dan one-hot encoding).

  4. Split data menjadi data latih dan uji.

  5. Training model Random Forest dan XGBoost.

  6. Evaluasi model dan visualisasi hasil.

# ✨ CUMLIPAN KODE UTAMA

📥 Preprocessing dan Pipeline

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), fitur_numerik),
    ('cat', OneHotEncoder(handle_unknown='ignore'), fitur_kategorikal)
])

🌳 Random Forest Pipeline

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

🚀 XGBoost Pipeline

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(objective='reg:squarederror'))
])

# 📊 EVALUASI PERFORMA
📌 Gambar: Visualisasi Prediksi Random Forest & XGBoost
![Screenshot 2025-07-08 222540](https://github.com/user-attachments/assets/5026bc5d-e7d6-48d4-aa5e-6d1fd266044e)

| Model         | MAE (Rp)    | RMSE (Rp)   | R² Score   |
| ------------- | ----------- | ----------- | --------   |
| Random Forest | 654319962   | 1860495807  | 0.8432     |
| XGBoost       | 637125215   | 1716557060  | 0.8665     |

# ✅ KESIMPULAN
  - Kedua model mampu memprediksi harga rumah dengan cukup baik.

  - XGBoost sedikit lebih unggul dari Random Forest dalam hal akurasi (R² lebih tinggi).

  - Fitur seperti lokasi (city, district), luas tanah, dan jumlah kamar terbukti sangat berpengaruh terhadap harga rumah.
# 🛠️ SARAN PENGEMBANGAN
  - Coba integrasikan data eksternal seperti harga tanah per wilayah atau fasilitas umum terdekat.

  - Gunakan SHAP values untuk interpretasi model lebih dalam.

  - Uji model dengan cross-validation k-fold untuk kestabilan performa.

  - Buat API prediksi sederhana agar bisa digunakan dalam website properti.
