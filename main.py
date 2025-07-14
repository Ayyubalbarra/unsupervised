import streamlit as st
import pandas as pd
import numpy as np
from clustering import load_and_clean_data, scale_features, perform_kmeans
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# ===== KONFIGURASI HALAMAN =====
st.set_page_config(
    page_title="Clustering Ekonomi Provinsi Indonesia",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', sans-serif;
    color: #f0f2f6;
}
.main-header {
    background: linear-gradient(90deg, #00416A, #E4E5E6);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}
.metric-container {
    background: #1c1c1c;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #4CAF50;
    margin: 1rem 0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    color: #fff;
}
.cluster-info {
    background: #2b2b2b;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #444;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    margin: 1rem 0;
    color: #f0f2f6;
}
.success-box {
    background: #1e4620;
    color: #b2fab4;
    border-left: 5px solid #2e7d32;
}
.warning-box {
    background: #4e3c1f;
    color: #ffe082;
    border-left: 5px solid #ffa000;
}
.error-box {
    background: #5f2120;
    color: #ef9a9a;
    border-left: 5px solid #e53935;
}
.stTabs [role="tablist"] > div[aria-selected="true"] {
    border-bottom: 3px solid #29b6f6;
    color: #29b6f6;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ===== FUNGSI =====
@st.cache_data
def load_data():
    return load_and_clean_data()

def format_currency(value):
    return f"Rp {value:,.0f}"

def create_cluster_interpretation(cluster_data, df_mean):
    pdrb_high = cluster_data['PDRB_per_kapita'] > df_mean['PDRB_per_kapita']
    ipm_high = cluster_data['IPM'] > df_mean['IPM']
    unemployment_low = cluster_data['Tingkat_Pengangguran'] < df_mean['Tingkat_Pengangguran']
    poverty_low = cluster_data['Kemiskinan'] < df_mean['Kemiskinan']

    if pdrb_high and ipm_high and unemployment_low and poverty_low:
        return "🟢 Provinsi Maju", "success", "PDRB tinggi, IPM tinggi, pengangguran & kemiskinan rendah"
    elif pdrb_high and ipm_high:
        return "🟡 Provinsi Berkembang Maju", "warning", "PDRB dan IPM tinggi, masih ada tantangan sosial"
    elif ipm_high:
        return "🔵 Provinsi Berkembang", "info", "IPM baik, perlu peningkatan ekonomi"
    else:
        return "🔴 Provinsi Tertinggal", "error", "Perlu perhatian khusus di semua aspek"

def create_plot(df, x, y, title):
    fig = px.scatter(
        df, x=x, y=y, color="Cluster",
        hover_data=["Provinsi", "PDRB_per_kapita", "IPM", "Tingkat_Pengangguran", "Kemiskinan"],
        title=title, color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(template="plotly_dark", height=500)
    return fig

# ===== HEADER =====
st.markdown("""
<div class="main-header">
    <h1>🏛️ Clustering Provinsi Indonesia</h1>
    <h3>Analisis Berdasarkan Indikator Ekonomi</h3>
</div>
""", unsafe_allow_html=True)

# ===== SIDEBAR =====
st.sidebar.header("⚙️ Pengaturan Clustering")
k = st.sidebar.slider("Jumlah Cluster", 2, 6, 3)
chart_type = st.sidebar.selectbox("Jenis Visualisasi", ["Plotly Interaktif", "Matplotlib Statik"])

# ===== DATA =====
with st.spinner("🔄 Memuat data..."):
    df = load_data()
if df is None or df.empty:
    st.error("❌ Gagal memuat data.")
    st.stop()

X_scaled, features = scale_features(df)
df['Cluster'] = perform_kmeans(X_scaled, k)

# ===== TABS =====
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🔍 Cluster", "📈 Visualisasi", "💡 Interpretasi"])

with tab1:
    st.subheader("📊 Dataset dan Statistik")
    st.dataframe(df.drop("Cluster", axis=1), use_container_width=True)

with tab2:
    st.subheader("🔍 Distribusi Cluster")
    for i in range(k):
        prov = df[df['Cluster'] == i]['Provinsi'].tolist()
        st.markdown(f"### Cluster {i} ({len(prov)} Provinsi)")
        st.markdown("<div class='cluster-info'><ul>" + "".join([f"<li>{p}</li>" for p in prov]) + "</ul></div>", unsafe_allow_html=True)

with tab3:
    st.subheader("📈 Visualisasi Data")
    if chart_type == "Plotly Interaktif":
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_plot(df, "PDRB_per_kapita", "IPM", "PDRB vs IPM"), use_container_width=True)
        with col2:
            st.plotly_chart(create_plot(df, "Kemiskinan", "Tingkat_Pengangguran", "Kemiskinan vs Pengangguran"), use_container_width=True)
        st.plotly_chart(px.scatter_3d(df, x='PDRB_per_kapita', y='IPM', z='Tingkat_Pengangguran', color='Cluster', hover_name='Provinsi', title='📊 3D Visualisasi Cluster'), use_container_width=True)
    else:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="PDRB_per_kapita", y="IPM", hue="Cluster", palette="viridis", s=100, ax=ax)
        st.pyplot(fig)

with tab4:
    st.subheader("💡 Interpretasi & Strategi")
    cluster_summary = df.groupby("Cluster")[features].mean()
    df_mean = df[features].mean()

    for i in range(k):
        cluster_data = cluster_summary.loc[i]
        prov = df[df['Cluster'] == i]['Provinsi'].tolist()
        status, box, desc = create_cluster_interpretation(cluster_data, df_mean)
        with st.expander(f"{status} - Cluster {i} ({len(prov)} Provinsi)"):
            st.metric("💰 PDRB per Kapita", format_currency(cluster_data['PDRB_per_kapita']))
            st.metric("🎓 IPM", f"{cluster_data['IPM']:.2f}")
            st.metric("👥 Pengangguran", f"{cluster_data['Tingkat_Pengangguran']:.2f}%")
            st.metric("🏠 Kemiskinan", f"{cluster_data['Kemiskinan']:.2f}%")

            box_class = "success-box" if box == "success" else "warning-box" if box == "warning" else "error-box"
            st.markdown(f"<div class='{box_class}'><strong>{desc}</strong><br><em>Strategi: Tindak lanjut sesuai kondisi masing-masing cluster.</em></div>", unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown("""
---
<div style='text-align:center; color: gray;'>
    Aplikasi oleh Tim Analitik | 2025
</div>
""", unsafe_allow_html=True)
