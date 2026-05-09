import numpy as np
from tslearn.metrics import cdist_dtw
import hdbscan

def run_clustering(df_sampled):
    pivot_df = df_sampled.pivot(index='id_lokasi', columns='tanggal', values='NDVI_smooth')
    pivot_df = pivot_df.ffill(axis=1).bfill(axis=1)

    data_3d = pivot_df.values[:, :, np.newaxis].astype(np.float32)
    
    # Hitung matriks jarak DTW
    dist_matrix = cdist_dtw(data_3d, n_jobs=-1, verbose=0)

    clusterer = hdbscan.HDBSCAN(
        metric='precomputed',
        min_cluster_size=3,
        min_samples=None,
        cluster_selection_epsilon=0,
        gen_min_span_tree=True,
        prediction_data=True
    )
    labels = clusterer.fit_predict(dist_matrix)

    unique_clusters = sorted(set(labels) - {-1})
    n_clusters = len(unique_clusters)
    n_noise = sum(1 for l in labels if l == -1)

    return labels, pivot_df.index.tolist(), n_clusters, n_noise