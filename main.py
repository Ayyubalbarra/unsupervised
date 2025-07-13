import streamlit as st
import pandas as pd
from clustering import load_and_clean_data, scale_features, perform_kmeans
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import io

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Clustering Ekonomi Provinsi", layout="wide")

# === JUDUL APLIKASI ===
st.title("📊 Clustering Provinsi di Indonesia Berdasarkan Indikator Ekonomi")
st.write("Aplikasi ini mengelompokkan provinsi berdasarkan PDRB per kapita, IPM, tingkat pengangguran, dan kemiskinan.")

# === STEP 1: Muat dan Tampilkan Data Awal ===
st.header("1. Dataset Awal")

# Capture debug output
debug_output = io.StringIO()
sys.stdout = debug_output

with st.spinner('Memuat data dari `data/final_dataset.csv`...'):
    try:
        df = load_and_clean_data()
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        # Show debug info
        debug_info = debug_output.getvalue()
        if debug_info:
            with st.expander("🔍 Debug Info - Klik untuk melihat detail pemrosesan"):
                st.code(debug_info)
        
        if df is None or len(df) == 0:
            st.error("❌ Gagal memuat data atau data kosong setelah pembersihan.")
            st.write("**Kemungkinan masalah:**")
            st.write("1. File CSV tidak ditemukan")
            st.write("2. Format data tidak sesuai")
            st.write("3. Semua data tidak valid/kosong")
            st.stop()
        
        st.dataframe(df, use_container_width=True)
        st.success(f"✅ Data berhasil dimuat dan dibersihkan. Terdapat **{len(df)}** provinsi yang valid untuk dianalisis.")
        
        # Tampilkan informasi statistik dasar
        st.subheader("📊 Statistik Deskriptif")
        numeric_cols = ['PDRB_per_kapita', 'Tingkat_Pengangguran', 'IPM', 'Kemiskinan']
        
        # Format statistik untuk tampilan yang lebih baik
        stats_df = df[numeric_cols].describe()
        
        # Format PDRB sebagai mata uang
        if 'PDRB_per_kapita' in stats_df.columns:
            pdrb_stats = stats_df['PDRB_per_kapita'].copy()
            stats_df['PDRB_per_kapita'] = pdrb_stats.apply(lambda x: f"Rp {x:,.0f}")
        
        st.dataframe(stats_df, use_container_width=True)
        
    except FileNotFoundError:
        sys.stdout = sys.__stdout__
        st.error("❌ Error: File `data/final_dataset.csv` tidak ditemukan.")
        st.write("**Solusi:**")
        st.write("1. Pastikan file ada di dalam folder `data/`")
        st.write("2. Pastikan nama file adalah `final_dataset.csv`")
        st.write("3. Periksa struktur folder proyek")
        st.stop()
    except Exception as e:
        sys.stdout = sys.__stdout__
        st.error(f"❌ Terjadi error saat memuat data: {e}")
        st.write("**Debug info:**")
        st.code(debug_output.getvalue())
        st.stop()

# === STEP 2: Proses Clustering ===
st.header("2. Hasil Clustering dengan K-Means")
st.write("Pilih jumlah cluster dan jalankan algoritma K-Means.")

# Pilihan jumlah cluster
k = st.slider("Pilih Jumlah Cluster (k):", min_value=2, max_value=min(5, len(df)), value=3)

# Capture debug output for clustering
debug_output_clustering = io.StringIO()
sys.stdout = debug_output_clustering

try:
    # Scaling dan clustering
    X_scaled, features = scale_features(df)
    clusters = perform_kmeans(X_scaled, k=k)
    
    # Restore stdout
    sys.stdout = sys.__stdout__
    
    # Show clustering debug info
    clustering_debug = debug_output_clustering.getvalue()
    if clustering_debug:
        with st.expander("🔍 Debug Info Clustering"):
            st.code(clustering_debug)
    
    # Tambahkan hasil cluster ke DataFrame utama
    df['Cluster'] = clusters
    
    # Tampilkan hasil dalam tabel yang ringkas
    st.write("📋 Hasil penetapan cluster untuk setiap provinsi:")
    result_df = df[['Provinsi', 'Cluster']].copy()
    result_df['Cluster'] = result_df['Cluster'].apply(lambda x: f"Cluster {x}")
    st.dataframe(result_df, use_container_width=True, height=300)
    
    # Tampilkan distribusi cluster
    cluster_counts = df['Cluster'].value_counts().sort_index()
    st.write("📊 Distribusi Cluster:")
    col1, col2, col3 = st.columns(3)
    for i, (cluster, count) in enumerate(cluster_counts.items()):
        with [col1, col2, col3][i % 3]:
            st.metric(f"Cluster {cluster}", f"{count} provinsi")
    
except Exception as e:
    sys.stdout = sys.__stdout__
    st.error(f"❌ Error dalam proses clustering: {e}")
    st.write("**Debug info:**")
    st.code(debug_output_clustering.getvalue())
    st.stop()

# === STEP 3: Visualisasi Hasil Clustering ===
st.header("3. 📈 Visualisasi Cluster")
st.write("Visualisasi sebaran provinsi berdasarkan indikator yang dipilih.")

# Set style untuk plot
plt.style.use('default')
sns.set_palette("viridis")

col1, col2 = st.columns(2)

with col1:
    # Scatter Plot: PDRB vs IPM
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    scatter = sns.scatterplot(data=df, x='PDRB_per_kapita', y='IPM', hue='Cluster', 
                   palette='viridis', s=120, ax=ax1, legend='full')
    ax1.set_title("Cluster: PDRB per Kapita vs IPM", fontsize=16, fontweight='bold')
    ax1.set_xlabel("PDRB per Kapita (Rupiah)", fontsize=12)
    ax1.set_ylabel("Indeks Pembangunan Manusia (IPM)", fontsize=12)
    
    # Format x-axis untuk nilai PDRB
    ax1.ticklabel_format(style='plain', axis='x')
    plt.setp(ax1.get_xticklabels(), rotation=45)
    
    # Tambahkan grid
    ax1.grid(True, alpha=0.3)
    
    st.pyplot(fig1)

with col2:
    # Scatter Plot: Kemiskinan vs Pengangguran
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=df, x='Kemiskinan', y='Tingkat_Pengangguran', hue='Cluster', 
                   palette='viridis', s=120, ax=ax2, legend='full')
    ax2.set_title("Cluster: Kemiskinan vs Pengangguran", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Tingkat Kemiskinan (%)", fontsize=12)
    ax2.set_ylabel("Tingkat Pengangguran (%)", fontsize=12)
    
    # Tambahkan grid
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig2)

# === STEP 4: Analisis dan Ringkasan Karakteristik Cluster ===
st.header("4. 📊 Ringkasan Karakteristik Setiap Cluster")
st.write("Tabel di bawah ini menunjukkan nilai rata-rata dari setiap indikator untuk masing-masing cluster.")

try:
    # Gunakan groupby untuk ringkasan
    cluster_summary = df.groupby('Cluster')[features].mean()
    
    # Format angka untuk tampilan yang lebih baik
    cluster_summary_formatted = cluster_summary.copy()
    cluster_summary_formatted['PDRB_per_kapita'] = cluster_summary_formatted['PDRB_per_kapita'].apply(
        lambda x: f"Rp {x:,.0f}"
    )
    for col in ['Tingkat_Pengangguran', 'IPM', 'Kemiskinan']:
        if col in cluster_summary_formatted.columns:
            cluster_summary_formatted[col] = cluster_summary_formatted[col].apply(
                lambda x: f"{x:.2f}%"
            )
    
    st.dataframe(cluster_summary_formatted, use_container_width=True)
    
    # Tampilkan anggota setiap cluster
    st.write("🏛️ **Daftar provinsi di setiap cluster:**")
    for i in range(k):
        with st.expander(f"**📍 Cluster {i} - Detail**"):
            provinces = df[df['Cluster'] == i]['Provinsi'].tolist()
            st.write(f"**Jumlah Provinsi:** {len(provinces)}")
            st.write(f"**Provinsi:** {', '.join(provinces)}")
            
            # Tampilkan karakteristik cluster
            cluster_data = df[df['Cluster'] == i][features].mean()
            st.write("**📈 Karakteristik Rata-rata:**")
            st.write(f"- 💰 PDRB per Kapita: Rp {cluster_data['PDRB_per_kapita']:,.0f}")
            st.write(f"- 🎓 IPM: {cluster_data['IPM']:.2f}")
            st.write(f"- 👥 Tingkat Pengangguran: {cluster_data['Tingkat_Pengangguran']:.2f}%")
            st.write(f"- 🏠 Kemiskinan: {cluster_data['Kemiskinan']:.2f}%")
            
            # Interpretasi cluster
            if cluster_data['PDRB_per_kapita'] > df['PDRB_per_kapita'].mean():
                if cluster_data['IPM'] > df['IPM'].mean():
                    st.success("✅ **Profil:** Provinsi Maju (PDRB dan IPM tinggi)")
                else:
                    st.warning("⚠️ **Profil:** Provinsi Kaya tapi IPM rendah")
            else:
                if cluster_data['IPM'] < df['IPM'].mean():
                    st.error("❌ **Profil:** Provinsi Tertinggal (perlu perhatian khusus)")
                else:
                    st.info("ℹ️ **Profil:** Provinsi Berkembang")
            
except Exception as e:
    st.error(f"❌ Error dalam analisis cluster: {e}")

# === STEP 5: Interpretasi Hasil ===
st.header("5. 🎯 Interpretasi Hasil & Rekomendasi")
st.write("""
**📋 Interpretasi Cluster:**

**🟢 Cluster Maju:** Provinsi dengan PDRB tinggi dan IPM tinggi
- Karakteristik: Ekonomi kuat, pembangunan manusia baik
- Strategi: Pertahankan momentum, fokus pada inovasi

**🟡 Cluster Berkembang:** Provinsi dengan karakteristik menengah
- Karakteristik: Dalam transisi pembangunan
- Strategi: Percepatan pembangunan infrastruktur dan SDM

**🔴 Cluster Tertinggal:** Provinsi dengan PDRB rendah dan IPM rendah
- Karakteristik: Memerlukan perhatian khusus
- Strategi: Program khusus pengentasan kemiskinan dan peningkatan pendidikan

**🎯 Manfaat Analisis:**
- 📊 Perencanaan alokasi anggaran yang lebih tepat sasaran
- 🎯 Penentuan prioritas program pembangunan
- 🗺️ Strategi pengembangan ekonomi regional yang berbeda per cluster
- 📈 Monitoring dan evaluasi kemajuan pembangunan
""")

# Footer
st.markdown("---")
st.markdown("*Aplikasi Clustering Provinsi Indonesia - Analisis Indikator Ekonomi*")
st.markdown("*Data diproses menggunakan algoritma K-Means*")