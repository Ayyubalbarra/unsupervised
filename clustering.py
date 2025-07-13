# clustering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import re

def load_and_clean_data():
    """
    Memuat data dari CSV, membersihkan, dan menangani nilai yang hilang.
    """
    # Baca file CSV dengan delimiter pipe
    df = pd.read_csv("data/final_dataset.csv", delimiter='|')
    
    # Debug: Tampilkan kolom yang tersedia
    print("Kolom asli:", df.columns.tolist())
    
    # Bersihkan nama kolom dari spasi ekstra
    df.columns = df.columns.str.strip()
    
    # Debug: Tampilkan kolom setelah dibersihkan
    print("Kolom setelah dibersihkan:", df.columns.tolist())
    
    # Cek kolom yang tersedia dan buat mapping yang tepat
    available_columns = df.columns.tolist()
    actual_column_mapping = {}
    
    # Mapping berdasarkan kolom yang sebenarnya tersedia
    for col in available_columns:
        if 'PDRB' in col:
            actual_column_mapping[col] = 'PDRB_per_kapita'
        elif 'Tingkat' in col and 'Pengangguran' in col:
            actual_column_mapping[col] = 'Tingkat_Pengangguran'
        elif 'IPM' in col:
            actual_column_mapping[col] = 'IPM'
        elif 'Kemiskinan' in col:
            actual_column_mapping[col] = 'Kemiskinan'
        elif 'Provinsi' in col:
            actual_column_mapping[col] = 'Provinsi'
    
    print("Mapping kolom yang akan digunakan:", actual_column_mapping)
    
    # Rename kolom menggunakan mapping yang tepat
    df = df.rename(columns=actual_column_mapping)
    
    # Hapus kolom yang tidak diperlukan (Unnamed columns)
    df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')
    
    # Debug: Tampilkan kolom final
    print("Kolom final:", df.columns.tolist())
    
    # Kolom yang akan digunakan untuk analisis
    numeric_cols = ['PDRB_per_kapita', 'Tingkat_Pengangguran', 'IPM', 'Kemiskinan']
    
    # Cek apakah semua kolom yang diperlukan ada
    missing_cols = [col for col in numeric_cols if col not in df.columns]
    if missing_cols:
        print(f"Kolom yang tidak ditemukan: {missing_cols}")
        return None
    
    # Fungsi untuk membersihkan data numerik
    def clean_numeric(value):
        if pd.isna(value):
            return None
        # Konversi ke string dan hapus karakter non-numerik kecuali titik dan minus
        value_str = str(value)
        # Hapus simbol %, koma, spasi, dan karakter lainnya
        cleaned = re.sub(r'[,%\s\+\-\(\)]', '', value_str)
        # Hapus karakter non-digit dan non-titik
        cleaned = re.sub(r'[^\d\.]', '', cleaned)
        try:
            return float(cleaned) if cleaned else None
        except ValueError:
            return None
    
    # Bersihkan dan konversi kolom numerik
    for col in numeric_cols:
        if col in df.columns:
            print(f"Membersihkan kolom: {col}")
            df[col] = df[col].apply(clean_numeric)
            print(f"Sample data {col}:", df[col].head().tolist())
        else:
            print(f"Kolom {col} tidak ditemukan dalam file CSV")
    
    # Hapus baris dengan nilai NaN di kolom-kolom penting
    before_clean = len(df)
    df.dropna(subset=numeric_cols, inplace=True)
    after_clean = len(df)
    
    print(f"Data sebelum pembersihan: {before_clean} baris")
    print(f"Data setelah pembersihan: {after_clean} baris")
    print(f"Baris yang dihapus: {before_clean - after_clean}")
    
    if len(df) == 0:
        print("Warning: Tidak ada data yang valid setelah pembersihan!")
        return None
    
    return df

def scale_features(df):
    """
    Memilih fitur dan melakukan penskalaan (standardization).
    """
    features = ['PDRB_per_kapita', 'Tingkat_Pengangguran', 'IPM', 'Kemiskinan']
    X = df[features]
    
    print("Data sebelum scaling:")
    print(X.describe())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Data berhasil di-scale")
    
    return X_scaled, features

def perform_kmeans(X_scaled, k=3):
    """
    Menjalankan algoritma K-Means untuk membuat cluster.
    """
    print(f"Menjalankan K-Means dengan k={k}")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    print(f"Clustering selesai. Distribusi cluster: {pd.Series(clusters).value_counts().sort_index().tolist()}")
    return clusters