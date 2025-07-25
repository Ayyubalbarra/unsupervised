# Clustering Provinsi di Indonesia Menggunakan K-Means

## Informasi Kelompok
- **Nama**: Ayyub Al Barra
- **NIM**: 23523104
- **Kelas**: B

- **Nama**: Rafi Fausta Ramadhan
- **NIM**: 23523196
- **Kelas**: B

## Deskripsi Proyek
Proyek ini merupakan implementasi dari algoritma *Unsupervised Learning*, yaitu **K-Means Clustering**, untuk mengelompokkan provinsi-provinsi di Indonesia berdasarkan indikator-indikator ekonomi dan sosial. Tujuannya adalah untuk memetakan dan menganalisis disparitas pembangunan antar-provinsi.

Hasil dari analisis clustering ini disajikan dalam bentuk **dasbor web interaktif** yang dibangun menggunakan **Streamlit**, memungkinkan visualisasi dan interpretasi data yang mudah dipahami.

**Teknologi yang Digunakan:**
- Python
- Pandas & NumPy untuk manipulasi data
- Scikit-learn untuk implementasi K-Means dan penskalaan data
- Streamlit untuk membangun dasbor interaktif
- Plotly & Matplotlib/Seaborn untuk visualisasi data

## Sumber Data
Data yang digunakan adalah data sekunder mengenai kondisi sosio-ekonomi provinsi di Indonesia. Data ini dikumpulkan dari berbagai sumber yang relevan (diasumsikan berasal dari Badan Pusat Statistik - BPS) dan merupakan data terbaru yang tersedia (maksimal 5 tahun terakhir, sesuai ketentuan tugas).

**Variabel Input (Fitur) yang Digunakan:**
1.  **PDRB per Kapita (`PDRB_per_kapita`)**: Menggambarkan pendapatan rata-rata penduduk di suatu provinsi.
2.  **Tingkat Pengangguran Terbuka (`Tingkat_Pengangguran`)**: Persentase angkatan kerja yang tidak memiliki pekerjaan.
3.  **Indeks Pembangunan Manusia (`IPM`)**: Indikator gabungan dari harapan hidup, tingkat pendidikan, dan standar hidup layak.
4.  **Tingkat Kemiskinan (`Kemiskinan`)**: Persentase penduduk yang hidup di bawah garis kemiskinan.


## Alur Program (Algoritma)
Alur kerja program dibagi menjadi dua file utama: `clustering.py` untuk logika backend dan `main.py` untuk frontend (dasbor).

1.  **Pemuatan dan Pembersihan Data (`clustering.py: load_and_clean_data`)**
    * Membaca dataset `final_dataset.csv` yang menggunakan delimiter `|`.
    * Membersihkan nama kolom dari spasi yang tidak diinginkan.
    * Melakukan pemetaan nama kolom untuk memastikan konsistensi.
    * Membersihkan data numerik dari karakter non-numerik (seperti '%', ',', spasi) menggunakan Regular Expressions (Regex).
    * Mengonversi kolom data ke tipe `float`.
    * Menghapus baris yang memiliki nilai kosong (`NaN`) pada kolom fitur untuk memastikan kualitas data.

2.  **Penskalaan Fitur (`clustering.py: scale_features`)**
    * Menggunakan `StandardScaler` dari Scikit-learn untuk melakukan standardisasi pada semua fitur numerik.
    * **Alasan**: K-Means adalah algoritma yang sensitif terhadap skala data. Penskalaan ini penting agar variabel dengan rentang nilai besar (misal: PDRB) tidak mendominasi proses clustering dan mengabaikan kontribusi variabel lain.

3.  **Implementasi K-Means (`clustering.py: perform_kmeans`)**
    * Menerapkan algoritma K-Means pada data yang telah diskalakan.
    * Jumlah cluster (`k`) ditentukan secara dinamis berdasarkan input dari pengguna di dasbor Streamlit (default `k=3`).
    * `random_state=42` digunakan untuk memastikan hasil clustering konsisten (reproducible) setiap kali kode dijalankan.
    * `n_init=10` digunakan untuk menjalankan algoritma 10 kali dengan inisialisasi centroid yang berbeda dan memilih hasil terbaik untuk menghindari local optima.

4.  **Visualisasi dan Interpretasi (`main.py`)**
    * Menggunakan Streamlit untuk membuat antarmuka pengguna yang interaktif.
    * Terdapat slider untuk memungkinkan pengguna memilih jumlah cluster (`k`) dari 2 hingga 6.
    * Hasil clustering divisualisasikan menggunakan:
        * **Plotly Scatter Plots (2D & 3D)**: Untuk menampilkan sebaran data dan cluster secara interaktif.
        * **Distribusi Anggota Cluster**: Menampilkan daftar provinsi untuk setiap cluster.
    * **Interpretasi Otomatis**: Dasbor memberikan label deskriptif pada setiap cluster (misal: "Provinsi Maju", "Provinsi Tertinggal") dengan membandingkan nilai rata-rata (centroid) cluster dengan rata-rata keseluruhan data.

## Analisis dan Temuan
(Bagian ini dapat diisi dengan ringkasan dari analisis komprehensif yang telah saya berikan di atas, mencakup poin 1-9 seperti visualisasi, pre-processing, pemilihan metode, evaluasi, temuan, dll.)

## Cara Menjalankan Proyek
1.  Pastikan Anda memiliki Python dan `pip` terinstal.
2.  Instal semua library yang dibutuhkan. Disarankan membuat virtual environment.
    ```bash
    pip install pandas scikit-learn streamlit seaborn matplotlib plotly
    ```
3.  Letakkan semua file (`main.py`, `clustering.py`, dan folder `data/` berisi `final_dataset.csv`) dalam satu direktori.
4.  Jalankan aplikasi Streamlit melalui terminal:
    ```bash
    streamlit run main.py
    ```
5.  Aplikasi akan terbuka secara otomatis di browser Anda.