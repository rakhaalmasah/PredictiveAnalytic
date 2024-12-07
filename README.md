# Laporan Proyek Machine Learning - [Muhammad Rakha Almasah]

## Domain Proyek

### **Latar Belakang**
Pasar saham adalah indikator vital kesehatan ekonomi suatu negara. Analisis dan prediksi pergerakan indeks saham, seperti NASDAQ dan S&P 500 (SPY), membantu investor dalam pengambilan keputusan yang lebih baik. Namun, dinamika pergerakan saham sangat kompleks, dipengaruhi oleh berbagai faktor, termasuk kinerja perusahaan besar seperti Apple, Microsoft, Amazon, dan Berkshire Hathaway.

Penelitian sebelumnya telah mengeksplorasi berbagai pendekatan untuk memprediksi pasar saham. Misalnya, Shah et al. (2021) menggunakan prediksi berbasis berita dengan text mining dan machine learning untuk meningkatkan akurasi prediksi pasar keuangan. Selain itu, Kim et al. (2020) menerapkan algoritma hybrid machine learning untuk mengidentifikasi arah pergerakan harian pasar saham dengan lebih akurat. Lebih lanjut, Huang et al. (2021) membahas aplikasi machine learning dalam peramalan keuangan, perencanaan, dan analisis, sementara Gupta et al. (2023) menganalisis pengaruh kecerdasan buatan dan machine learning terhadap pasar keuangan, termasuk dalam trading, manajemen risiko, dan operasi keuangan.

Prediksi indeks saham dengan machine learning menawarkan potensi besar dalam memberikan wawasan bagi investor dan analis keuangan untuk memahami pola pasar yang kompleks.

### **Referensi**
- [News-Based Intelligent Prediction of Financial Markets Using Text Mining and Machine Learning: A Systematic Literature Review](https://scholar.google.com/scholar?hl=id&as_sdt=0%2C5&q=News-Based+Intelligent+Prediction+of+Financial+Markets+Using+Text+Mining+and+Machine+Learning%3A+A+Systematic+Literature+Review&btnG=)

- [Predicting the Daily Return Direction of the Stock Market Using Hybrid Machine Learning Algorithms](https://scholar.google.com/scholar?hl=id&as_sdt=0%2C5&q=Predicting+the+Daily+Return+Direction+of+the+Stock+Market+Using+Hybrid+Machine+Learning+Algorithms+&btnG=)
  
- [Machine learning for financial forecasting, planning and analysis](https://link.springer.com/article/10.1007/s42521-021-00046-2)

- [Unveiling the Influence of Artificial Intelligence and Machine Learning on Financial Markets: A Comprehensive Analysis of AI Applications in Trading, Risk Management, and Financial Operations](https://www.mdpi.com/1911-8074/16/10/434)

  
## Business Understanding

### **Problem Statements**
1. Bagaimana membangun model machine learning untuk memprediksi nilai indeks saham NASDAQ dan SPY berdasarkan data historis perusahaan besar?
2. Algoritma machine learning apa yang memberikan prediksi paling akurat?
3. Bagaimana cara meningkatkan akurasi prediksi melalui teknik hyperparameter tuning?

### **Goals**
1. Mengembangkan model machine learning untuk memprediksi NASDAQ dan SPY berdasarkan data saham Apple, Microsoft, Amazon, dan Berkshire Hathaway.
2. Membandingkan performa algoritma seperti Random Forest, XGBoost, dan LightGBM dalam memprediksi nilai indeks saham.
3. Melakukan hyperparameter tuning untuk meningkatkan akurasi model.

### **Solution Statements**
1. Menggunakan berbagai algoritma seperti Random Forest, XGBoost, LightGBM, Gradient Boosting, dan CatBoost untuk memprediksi NASDAQ dan SPY.
2. Melakukan evaluasi model menggunakan metrik Mean Squared Error (MSE).
3. Meningkatkan performa model melalui GridSearchCV untuk menemukan hyperparameter terbaik.

---

## Data Understanding

### **Informasi Dataset**
Dataset mencakup data historis saham NASDAQ (target), SPY (target), Apple, Microsoft, Amazon, dan Berkshire Hathaway (fitur). Data diperoleh dari [Yahoo Finance](https://finance.yahoo.com/).

**Fitur-fitur yang tersedia:**
- **Date:** Tanggal data diambil.
- **Close_nasdaq:** Nilai penutupan NASDAQ (target 1).
- **Close_spy:** Nilai penutupan S&P 500 ETF (target 2).
- **Close_aapl:** Nilai penutupan Apple (fitur).
- **Close_msft:** Nilai penutupan Microsoft (fitur).
- **Close_amzn:** Nilai penutupan Amazon (fitur).
- **Close_brkb:** Nilai penutupan Berkshire Hathaway (fitur).

### **Exploratory Data Analysis**
- **Univariate Analysis:** Menunjukkan distribusi setiap fitur dan target menggunakan histogram.
- **Multivariate Analysis:** Korelasi antar fitur dan target menunjukkan hubungan positif yang kuat, dengan NASDAQ sangat berkorelasi dengan SPY (0.99).
- **Outliers:** Deteksi menggunakan boxplot, dan outliers dihapus untuk meningkatkan kualitas data.

---

## Data Preparation

### **Proses Data Preparation**
1. **Pemisahan Target dan Fitur:** Target: `Close_nasdaq` dan `Close_spy`. Fitur: `Close_aapl`, `Close_msft`, `Close_amzn`, dan `Close_brkb`.
2. **Split Dataset:** Dataset dibagi menjadi 90% data latih dan 10% data uji menggunakan `train_test_split`.
3. **Standarisasi Data:** Fitur numerik distandarisasi menggunakan `StandardScaler` untuk meningkatkan performa algoritma.

### **Alasan Data Preparation**
- **Pemisahan target dan fitur** memastikan bahwa model hanya menggunakan data input untuk prediksi.
- **Standarisasi data** diperlukan untuk algoritma yang sensitif terhadap skala fitur, seperti SVR.

---

## Modeling

### **Algoritma yang Digunakan**
1. **Random Forest:** Algoritma ensemble berbasis pohon keputusan yang robust terhadap overfitting.
2. **AdaBoost:** Memperbaiki kesalahan model sebelumnya secara iteratif.
3. **SVR:** Algoritma berbasis Support Vector Machine untuk regresi.
4. **Gradient Boosting:** Teknik boosting untuk menangkap pola non-linear.
5. **XGBoost:** Implementasi Gradient Boosting yang dioptimalkan.
6. **LightGBM:** Model boosting yang efisien untuk dataset besar.
7. **CatBoost:** Model boosting yang menangani data kategori dengan baik.
8. **ElasticNet:** Kombinasi regularisasi L1 dan L2 untuk regresi linier.

### **Hyperparameter Tuning**
Setiap model dioptimalkan menggunakan `GridSearchCV`. Parameter seperti `n_estimators`, `learning_rate`, dan `max_depth` diuji untuk menemukan kombinasi terbaik.

### **Hasil Evaluasi Model**
| Model             | Train MSE | Test MSE | Best Params                                |
|--------------------|-----------|----------|-------------------------------------------|
| Random Forest      | 0.337     | 1.952    | {'max_depth': 20, 'n_estimators': 150}    |
| XGBoost            | 0.363     | 2.337    | {'learning_rate': 0.2, 'n_estimators': 100}|
| LightGBM           | 0.832     | 2.113    | {'learning_rate': 0.2, 'n_estimators': 100}|
| Gradient Boosting  | 2.455     | 4.214    | {'learning_rate': 0.2, 'n_estimators': 100}|
| CatBoost           | 2.321     | 3.319    | {'iterations': 150, 'learning_rate': 0.2} |
| AdaBoost           | 32.244    | 35.258   | {'learning_rate': 0.2, 'n_estimators': 100}|
| SVR                | 30.100    | 31.615   | {'C': 10, 'kernel': 'linear'}             |
| ElasticNet         | 29.995    | 31.559   | {'alpha': 0.1, 'l1_ratio': 0.5}           |

- **Model Terbaik:** Random Forest dengan Test MSE terendah (1.952).

---

## Evaluation

### **Metrik Evaluasi**
Metrik yang digunakan adalah **Mean Squared Error (MSE)**:
\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
\]

- **MSE pada Random Forest:**
  - **Train MSE:** 0.337
  - **Test MSE:** 1.952

### **Visualisasi Prediksi**
1. **Prediksi NASDAQ:** Prediksi sangat dekat dengan nilai aktual, menunjukkan performa model yang sangat baik.
2. **Prediksi SPY:** Model juga memprediksi nilai SPY dengan akurasi tinggi, sebagaimana ditunjukkan oleh scatter plot.

### **Kesimpulan**
- Model Random Forest berhasil memprediksi NASDAQ dan SPY dengan akurasi tinggi, menjadikannya model terbaik untuk proyek ini.
- Hyperparameter tuning berkontribusi dalam meningkatkan akurasi model, terutama dibandingkan dengan model baseline.

---

## Catatan Tambahan
- **Peluang Perbaikan:**
  - Penambahan fitur lain yang relevan (contoh: data ekonomi makro).
  - Eksplorasi algoritma lain, seperti deep learning, untuk dataset yang lebih besar.
- **Implementasi:** Model dapat digunakan sebagai alat prediksi untuk membantu investor memahami pola pasar saham.

