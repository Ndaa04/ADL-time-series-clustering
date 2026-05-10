import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import re
import random


# ── HELPER ────────────────────────────────────────────────────

def _natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split('([0-9]+)', s)]


def assign_status_column(df, pivot_df):
    cluster_mapping = pivot_df['cluster'].to_dict()
    df = df.copy()
    df['cluster_id'] = df['id_lokasi'].map(cluster_mapping).fillna(-2)

    def _label(row):
        if row['cluster_id'] == -2: return 'Unselected'
        if row['cluster_id'] == -1: return 'Noise'
        return f"Cluster {int(row['cluster_id'])}"

    df['status'] = df.apply(_label, axis=1)
    return df


def get_valid_statuses(df):
    return sorted(
        [s for s in df['status'].unique() if s.startswith('Cluster')],
        key=_natural_sort_key
    )


# ── STEP 1: GRID PREVIEW (semua data, sebelum pilih tahun) ───

def plot_grid_preview(df, nr, nc, title='Sebaran Lokasi — Seluruh Data'):
    """Peta seluruh lokasi yang punya data (hijau) vs kosong (abu)."""
    grid = np.zeros((nr, nc), dtype=int)
    for _, row in df[['grid_row', 'grid_col']].drop_duplicates().iterrows():
        grid[int(row['grid_row']), int(row['grid_col'])] = 1

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.pcolormesh(np.arange(nc + 1), np.arange(nr + 1), grid,
                  cmap=ListedColormap(['#D3D3D3', '#4CAF50']),
                  edgecolors='white', linewidth=0.4, shading='auto')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.legend(handles=[
        Patch(facecolor='#4CAF50', edgecolor='white', label='Ada Data'),
        Patch(facecolor='#D3D3D3', edgecolor='white', label='Kosong'),
    ], loc='upper right')
    plt.tight_layout()
    return fig


# ── STEP 3: SMOOTHING PREVIEW ────────────────────────────────

def plot_smoothing_preview(df_smooth, n=3):
    """
    Perbandingan NDVI asli (abu transparan) vs NDVI smooth (biru solid)
    untuk n lokasi acak — ditampilkan setelah smoothing selesai.
    """
    locs = df_smooth['id_lokasi'].unique()
    preview = random.sample(list(locs), k=min(n, len(locs)))
    fig, axes = plt.subplots(1, len(preview), figsize=(5 * len(preview), 4), sharey=True)
    if len(preview) == 1:
        axes = [axes]
    for ax, loc in zip(axes, preview):
        sub = df_smooth[df_smooth['id_lokasi'] == loc].sort_values('tanggal')
        ax.plot(sub['tanggal'], sub['NDVI'],        color='#888888', alpha=0.35,
                linewidth=1.2, label='NDVI Asli')
        ax.plot(sub['tanggal'], sub['NDVI_smooth'], color='#1565C0',
                linewidth=2.2, label='Smoothed')
        ax.set_title(loc, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=30, labelsize=7)
    axes[0].set_ylabel('NDVI')
    axes[0].legend(loc='upper right', fontsize=7)
    plt.suptitle('NDVI Asli vs Savitzky‑Golay Smooth', fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ── STEP 4: SAMPLE GRID (hasil konfirmasi canvas) ────────────

def plot_sample_grid(df_year, sampled_ids, nr, nc):
    """
    Peta sebaran sampel — bentuk grid sama persis dengan plot_grid_preview.
    Warna:
      0 = kosong        → abu (#D3D3D3)
      1 = tidak dipilih → hijau transparan (alpha rendah)
      2 = tersampling   → biru (#2196F3)
    """
    sampled_set = set(sampled_ids)
    id_to_pos = df_year[['id_lokasi', 'grid_row', 'grid_col']].drop_duplicates('id_lokasi')

    grid = np.zeros((nr, nc), dtype=int)
    for _, row in id_to_pos.iterrows():
        grid[int(row['grid_row']), int(row['grid_col'])] = \
            2 if row['id_lokasi'] in sampled_set else 1

    # RGBA untuk hijau sangat transparan
    color_unsel = (0.30, 0.75, 0.30, 0.25)   # hijau pudar
    color_samp  = '#2196F3'                    # biru solid

    cmap = ListedColormap(['#D3D3D3', color_unsel, color_samp])

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.pcolormesh(np.arange(nc + 1), np.arange(nr + 1), grid,
                  cmap=cmap, vmin=0, vmax=2,
                  edgecolors='white', linewidth=0.3, shading='auto')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(f'Sebaran Sampel — {len(sampled_ids):,} Lokasi Terpilih',
                 fontweight='bold', fontsize=13)
    ax.legend(handles=[
        Patch(facecolor=color_samp,   edgecolor='white', label=f'Tersampling ({len(sampled_ids):,})'),
        Patch(facecolor=color_unsel,  edgecolor='white', label='Tidak Tersampling'),
        Patch(facecolor='#D3D3D3',    edgecolor='white', label='Kosong'),
    ], loc='upper right')
    plt.tight_layout()
    return fig


def plot_sample_ts_preview(df_year, sampled_ids, n=3):
    """Time series NDVI_smooth untuk n lokasi sampel acak."""
    preview = random.sample(list(sampled_ids), k=min(n, len(sampled_ids)))
    fig, axes = plt.subplots(1, len(preview), figsize=(5 * len(preview), 4), sharey=True)
    if len(preview) == 1:
        axes = [axes]
    colors = sns.color_palette('tab10', len(preview))
    for ax, loc_id, color in zip(axes, preview, colors):
        sub = df_year[df_year['id_lokasi'] == loc_id].sort_values('tanggal')
        ax.plot(sub['tanggal'], sub['NDVI_smooth'], color=color, linewidth=1.8)
        ax.fill_between(sub['tanggal'], sub['NDVI_smooth'], alpha=0.15, color=color)
        ax.set_title(loc_id, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=30, labelsize=7)
    axes[0].set_ylabel('NDVI Smoothed')
    plt.suptitle(f'Preview NDVI — {len(preview)} Lokasi Sampel', fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ── STEP 7: HASIL ────────────────────────────────────────────

def calculate_metrics(df, pivot_df):
    df = assign_status_column(df, pivot_df)
    valid_statuses = get_valid_statuses(df)

    df_clusters = df[df['cluster_id'] >= 0].copy()
    if df_clusters.empty:
        return pd.DataFrame(), pd.DataFrame(), [], df

    cluster_ts = df_clusters.groupby(['status', 'tanggal'])['NDVI_smooth'].agg(
        ['mean', 'std']
    ).reset_index()

    rows = []
    for s in valid_statuses:
        sub = cluster_ts[cluster_ts['status'] == s]
        if sub.empty: continue
        peak_idx = sub['mean'].idxmax()
        rows.append({
            'Status': s,
            'Jml Titik': df_clusters[df_clusters['status'] == s]['id_lokasi'].nunique(),
            'Puncak NDVI': sub['mean'].max(),
            'Waktu Puncak': sub.loc[peak_idx, 'tanggal'].date(),
            'Min NDVI': sub['mean'].min(),
            'Amplitudo': sub['mean'].max() - sub['mean'].min(),
            'Rata-rata NDVI': sub['mean'].mean(),
            'Rata-rata StdDev': sub['std'].mean(),
        })

    df_fenologi = pd.DataFrame(rows).set_index('Status') if rows else pd.DataFrame()
    return df_fenologi, cluster_ts, valid_statuses, df


def plot_comparison(cluster_ts, valid_statuses, tahun):
    palette = sns.color_palette('tab10', len(valid_statuses))
    colors = {s: palette[i] for i, s in enumerate(valid_statuses)}
    y_min = (cluster_ts['mean'] - cluster_ts['std']).min() - 0.05
    y_max = (cluster_ts['mean'] + cluster_ts['std']).max() + 0.05

    fig, ax = plt.subplots(figsize=(14, 6))
    for s in valid_statuses:
        sub = cluster_ts[cluster_ts['status'] == s]
        ax.plot(sub['tanggal'], sub['mean'], label=s, color=colors[s], linewidth=2.5)
        ax.fill_between(sub['tanggal'], sub['mean'] - sub['std'], sub['mean'] + sub['std'],
                        color=colors[s], alpha=0.1)
    ax.set_title(f'Perbandingan Tren NDVI Antar Cluster ({tahun})', fontsize=14, fontweight='bold')
    ax.set_ylabel('NDVI Smoothed')
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.subplots_adjust(right=0.82)
    return fig


def plot_individual_clusters(cluster_ts, valid_statuses):
    palette = sns.color_palette('tab10', len(valid_statuses))
    colors = {s: palette[i] for i, s in enumerate(valid_statuses)}
    n_cols = 2
    n_rows = max(1, (len(valid_statuses) + 1) // n_cols)
    y_min = (cluster_ts['mean'] - cluster_ts['std']).min() - 0.05
    y_max = (cluster_ts['mean'] + cluster_ts['std']).max() + 0.05

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for i, s in enumerate(valid_statuses):
        sub = cluster_ts[cluster_ts['status'] == s]
        axes[i].plot(sub['tanggal'], sub['mean'], color=colors[s], linewidth=2.5)
        axes[i].fill_between(sub['tanggal'], sub['mean'] - sub['std'], sub['mean'] + sub['std'],
                             color=colors[s], alpha=0.15)
        axes[i].set_title(f'Detail: {s}', fontsize=12, fontweight='bold')
        axes[i].set_ylim(y_min, y_max)
        axes[i].grid(True, alpha=0.2, linestyle='--')
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout(pad=3.5)
    return fig


def plot_spatial_map(df, nr, nc, valid_statuses, sampled_ids=None):
    """
    Peta akhir cluster:
      - Tidak tersampling (semua lokasi di luar pivot_df) → hijau pudar
      - Noise (cluster = -1)  → merah
      - Setiap cluster         → warna tab10
      - Kosong (tidak ada data)→ abu
    sampled_ids dipakai untuk membedakan 'tidak tersampling' vs 'kosong'.
    """
    NOISE_COLOR  = '#E74C3C'
    EMPTY_COLOR  = '#D3D3D3'
    UNSEL_COLOR  = (0.30, 0.75, 0.30, 0.30)   # hijau sangat transparan

    palette = sns.color_palette('tab10', len(valid_statuses))

    # integer map: 0=kosong, 1=unselected, 2=noise, 3+=cluster
    status_int = {'Unselected': 1, 'Noise': 2}
    for i, s in enumerate(valid_statuses):
        status_int[s] = i + 3

    # Tandai dulu semua sel yang ada lokasi sebagai 'Unselected'
    grid = np.zeros((nr, nc), dtype=int)
    all_pos = df[['grid_row', 'grid_col', 'status']].drop_duplicates()

    # Kalau sampled_ids diberikan, sel yang tidak masuk sampling tetap 0 (kosong visual)
    # vs sel yang disampling tapi noise/cluster
    sampled_set = set(sampled_ids) if sampled_ids is not None else None

    for _, row in all_pos.iterrows():
        r, c, st = int(row['grid_row']), int(row['grid_col']), row['status']
        val = status_int.get(st, 0)
        # Unselected = ada data tapi tidak masuk sampling → hijau pudar
        # Kosong     = tidak ada data sama sekali         → abu (0, default)
        grid[r, c] = val

    # Build colormap: [empty, unsel, noise, cluster0, cluster1, ...]
    n_total = 3 + len(valid_statuses)
    rgba_colors = (
        [EMPTY_COLOR, UNSEL_COLOR, NOISE_COLOR] +
        [mpl.colors.rgb2hex(palette[i]) for i in range(len(valid_statuses))]
    )
    cmap = ListedColormap(rgba_colors)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.pcolormesh(np.arange(nc + 1), np.arange(nr + 1), grid,
                  cmap=cmap, vmin=0, vmax=n_total - 1,
                  edgecolors='white', linewidth=0.3, shading='auto')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title('Peta Sebaran Cluster', fontweight='bold', fontsize=14)

    legend_els = (
        [Patch(facecolor=UNSEL_COLOR,  edgecolor='white', label='Tidak Tersampling'),
         Patch(facecolor=NOISE_COLOR,  edgecolor='white', label='Noise')] +
        [Patch(facecolor=mpl.colors.rgb2hex(palette[i]), edgecolor='white', label=s)
         for i, s in enumerate(valid_statuses)]
    )
    ax.legend(handles=legend_els, title='Keterangan',
              bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.75)
    return fig
