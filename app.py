import streamlit as st
import pandas as pd
import seaborn as sns
import gc

from data_loader import load_and_preprocess_data, smooth_and_resample, filter_and_sample_data
from clustering_engine import run_clustering
from visualizer import plot_time_series, plot_boxplot, plot_spatial_map

# Konfigurasi Halaman
st.set_page_config(page_title="NDVI Cluster Analysis", layout="wide")
st.title("🌿 Analisis Klaster NDVI Lamongan")

# Konstanta & Warna
NOISE_COLOR = '#E74C3C'
UNSELECTED_COLOR = '#9E9E9E'
VALID_PALETTE = sns.color_palette('tab10', 10)
TARGET_YEAR = 2024
N_SAMPLE = 500

# Cache Data Pipeline
@st.cache_data(ttl=3600)
def load_data_pipeline():
    df_clean = load_and_preprocess_data('Data_NDVI_Lamongan_2023.csv', 'Data_NDVI_Lamongan_2024.csv')
    df_smooth = smooth_and_resample(df_clean)
    df_year, df_sample, sample_locs = filter_and_sample_data(df_smooth, TARGET_YEAR, N_SAMPLE)
    return df_smooth, df_year, df_sample, sample_locs

# Cache Clustering
@st.cache_data(ttl=3600)
def run_clustering_pipeline(df_sample):
    labels, sampled_ids, n_clusters, n_noise = run_clustering(df_sample)
    return labels, sampled_ids

def main():
    st.sidebar.header("⚙️ Pengaturan Tampilan")
    show_time_series = st.sidebar.checkbox("📈 Grafik Time Series", value=True)
    show_boxplot = st.sidebar.checkbox("📦 Boxplot Distribusi", value=True)
    show_spatial = st.sidebar.checkbox("🗺️ Peta Spasial", value=True)

    with st.spinner("⏳ Memuat dan memproses data..."):
        df_smooth, df_year, df_sample, sample_locs = load_data_pipeline()

    with st.spinner("⏳ Menjalankan klastering HDBSCAN & DTW..."):
        labels, sampled_ids = run_clustering_pipeline(df_sample)

    # Mapping cluster ke dataframe utama
    cluster_map = pd.Series(labels, index=sampled_ids)
    df_final = df_year.copy()
    df_final['cluster_id'] = df_final['id_lokasi'].map(cluster_map)
    df_final['cluster_id'] = df_final['cluster_id'].fillna(-1).astype('int16')

    df_final['status'] = df_final.apply(
        lambda row: 'unselected' if row['id_lokasi'] not in sampled_ids
                    else ('noise' if row['cluster_id'] == -1 else f"Cluster {int(row['cluster_id'])}"),
        axis=1
    )

    valid_statuses = sorted(s for s in df_final['status'].unique() if s.startswith('Cluster'))
    plot_order_status = ['unselected', 'noise'] + valid_statuses

    STATUS_COLORS = {'unselected': UNSELECTED_COLOR, 'noise': NOISE_COLOR}
    for i, s in enumerate(valid_statuses):
        STATUS_COLORS[s] = VALID_PALETTE[i % len(VALID_PALETTE)]

    st.success(f"✅ Proses selesai. {df_final['status'].nunique()} kelompok ditemukan.")
    st.dataframe(df_final.groupby('status')['grid_id'].nunique().rename("Jumlah Grid"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if show_time_series:
            fig_ts = plot_time_series(df_final, TARGET_YEAR, STATUS_COLORS, plot_order_status)
            st.pyplot(fig_ts)
    with col2:
        if show_boxplot:
            fig_box = plot_boxplot(df_final, STATUS_COLORS, plot_order_status)
            st.pyplot(fig_box)

    if show_spatial:
        st.subheader("🗺️ Peta Spasial")
        fig_map = plot_spatial_map(df_final, valid_statuses, NOISE_COLOR, UNSELECTED_COLOR, VALID_PALETTE)
        st.pyplot(fig_map)

    st.subheader("📊 Ringkasan Statistik")
    summary = df_final.groupby('status')['NDVI_smooth'].agg(
        count='count', mean='mean', median='median', std='std', min='min', max='max'
    ).round(3)
    total_grids = df_final['grid_id'].nunique()
    summary['grid_count'] = df_final.groupby('status')['grid_id'].nunique()
    summary['area_pct'] = (summary['grid_count'] / total_grids * 100).round(1)

    st.dataframe(summary.style.format({
        'mean': '{:.3f}', 'median': '{:.3f}', 'std': '{:.3f}',
        'min': '{:.3f}', 'max': '{:.3f}', 'area_pct': '{:.1f}%',
        'count': '{:,.0f}', 'grid_count': '{:,.0f}'
    }).background_gradient(cmap='RdYlGn', subset=['mean']), use_container_width=True)

if __name__ == "__main__":
    main()