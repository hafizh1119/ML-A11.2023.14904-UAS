
# Analisis Komparatif Algoritma Machine Learning untuk Prediksi Harga Rumah Berdasarkan Dataset Jabodetabek

## Permasalahan
Bagaimana membangun model prediksi harga rumah yang akurat, efisien, dan dapat diandalkan, serta menentukan model mana yang paling baik berdasarkan evaluasi komparatif antar algoritma.

## Tujuan
- Mengevaluasi performa berbagai algoritma regresi dalam memprediksi harga rumah.
- Mengidentifikasi model dengan akurasi terbaik berdasarkan metrik RMSE dan R².
- Melihat seberapa baik model merepresentasikan data aktual secara visual dan numerik.
- Melakukan evaluasi mendalam terhadap pola kesalahan (residual) dari model terbaik.

## Dataset
Dataset berisi informasi harga rumah di wilayah Jabodetabek yang diekstrak dari file ZIP dan dimuat ke dalam DataFrame.

## 1. Import Library
Menggunakan `pandas`, `sklearn`, `matplotlib`, dan `seaborn` untuk manipulasi data, pemodelan, dan visualisasi.

## 2. Load Dataset
Membaca dataset dari file CSV, mengecek missing value, statistik deskriptif, dan distribusi kota.

## 3. Tentukan Target dan Fitur
- Target (`y`): Kolom `price_in_rp` (dikonversi logaritmik).
- Fitur (`X`): Semua kolom kecuali `price_in_rp`.

## 4. Pisahkan Fitur Numerik dan Kategorikal
Identifikasi kolom numerik dan kategorikal untuk preprocessing.

## 5. Buat Pipeline Preprocessing
- Numerik: Imputasi dengan median.
- Kategorikal: Imputasi dengan modus dan One-Hot Encoding.

## 6. Gabungkan dengan Random Forest
Membuat pipeline gabungan preprocessing dan model Random Forest.

## 7. Split Data
Membagi data menjadi 80% training dan 20% testing.

## 8. Latih Model
Model dilatih menggunakan data training.

## 9. Evaluasi Model
Evaluasi menggunakan MSE dan R². Hasil menunjukkan bahwa Random Forest memberikan hasil terbaik.

## 10. Feature Importance
Menampilkan 10 fitur paling berpengaruh, di mana `building_size_m2` adalah fitur dominan.

## 11. Visualisasi Fitur Terpenting
Menampilkan barplot 10 fitur paling berpengaruh terhadap harga rumah.

## 12. Eksperimen Model Lain
Model lain yang diuji:
- Random Forest
- Gradient Boosting
- Linear Regression
- Support Vector Regressor (SVR)

### Hasil Evaluasi:
- **Random Forest**: MSE 0.1087, R² 0.9127
- **Gradient Boosting**: MSE 0.1153, R² 0.9074
- **Linear Regression**: Performa sedang
- **SVR**: Performa terendah (underfitting)

## 13. Prediksi vs Asli
Visualisasi scatter plot menunjukkan bahwa Random Forest dan Gradient Boosting memberikan prediksi paling mendekati nilai aktual.

## 14. Analisis Error Model Terbaik
Residual dari Random Forest menunjukkan error tersebar merata di sekitar nol, menunjukkan model stabil dan tidak bias.

## 15. Distribusi Error Tiap Model
Distribusi error yang sempit dan simetris ditemukan pada Random Forest. SVR menunjukkan distribusi error terburuk.

## Kesimpulan
Random Forest menjadi model terbaik dalam prediksi harga rumah pada dataset Jabodetabek, dilihat dari evaluasi metrik dan analisis visual. Fitur `building_size_m2` menjadi penentu utama harga rumah. Model ini dapat digunakan sebagai dasar dalam sistem rekomendasi harga properti yang lebih akurat di masa depan.
