import pandas as pd
import numpy as np
from tslearn.metrics import cdist_dtw
import hdbscan


def run_clustering(df_sampled, min_cluster_size=3, min_samples=2,
                   cluster_selection_epsilon=0.05, progress_callback=None):
    """DTW + HDBSCAN pada df_sampled (sudah difilter tahun & sampling)."""
    if progress_callback: progress_callback(0.05, "📐 Pivoting time series...")
    pivot_df = df_sampled.pivot(index='id_lokasi', columns='tanggal', values='NDVI_smooth')
    pivot_df = pivot_df.ffill(axis=1).bfill(axis=1)

    if progress_callback: progress_callback(0.12, f"⏳ Menghitung DTW untuk {len(pivot_df)} lokasi...")
    data_3d = pivot_df.values[:, :, np.newaxis].astype(np.float32)
    dist_matrix = cdist_dtw(data_3d, n_jobs=-1, verbose=0)

    if progress_callback: progress_callback(0.80, "🔍 Menjalankan HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        metric='precomputed',
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        gen_min_span_tree=True,
        prediction_data=True
    )
    labels = clusterer.fit_predict(dist_matrix.astype(np.float64))

    if progress_callback: progress_callback(1.0, "✅ Clustering selesai!")
    pivot_df['cluster'] = labels
    return pivot_df


def get_dtw_description(df_sampled):
    """Statistik ringkas sebelum DTW dijalankan."""
    n_lok = df_sampled['id_lokasi'].nunique()
    n_ts  = df_sampled['tanggal'].nunique()
    return {
        'Jumlah Lokasi': f"{n_lok:,}",
        'Jumlah Timestamp': f"{n_ts:,}",
        'Periode': f"{df_sampled['tanggal'].min().date()} → {df_sampled['tanggal'].max().date()}",
        'Shape Matriks DTW': f"{n_lok} × {n_lok}",
        'Estimasi Ukuran': f"~{n_lok**2 * 8 / 1e6:.1f} MB",
    }