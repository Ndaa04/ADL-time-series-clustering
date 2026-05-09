import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import pandas as pd

def plot_time_series(df, target_year, STATUS_COLORS, plot_order_status):
    cluster_ts = df.groupby(['status', 'tanggal'])['NDVI_smooth'].agg(['mean', 'std']).reset_index()
    fig, ax = plt.subplots(figsize=(14, 6))
    x_min = pd.Timestamp(f'{target_year}-01-01')
    x_max = pd.Timestamp(f'{target_year}-12-31')

    for s in plot_order_status:
        sub = cluster_ts[cluster_ts['status'] == s]
        if sub.empty: continue
        color = STATUS_COLORS[s]
        label = 'Unselected' if s == 'unselected' else ('Noise' if s == 'noise' else s)
        ax.plot(sub['tanggal'], sub['mean'], label=label, color=color, linewidth=2)
        ax.fill_between(sub['tanggal'], sub['mean'] - sub['std'], sub['mean'] + sub['std'],
                        color=color, alpha=0.15)

    ax.set_xlim(x_min, x_max)
    ax.set_title(f'Tren NDVI Harian per Kelompok ({target_year})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('NDVI Smoothed')
    ax.set_ylim(-0.2, 1.0)
    ax.legend(title='Kelompok', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()
    return fig

def plot_boxplot(df, STATUS_COLORS, box_order):
    box_colors = [STATUS_COLORS[s] for s in box_order if s in df['status'].values]
    box_order = [s for s in box_order if s in df['status'].values]
    box_labels = ['Unselected' if s == 'unselected' else ('Noise' if s == 'noise' else s) for s in box_order]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x='status', y='NDVI_smooth', order=box_order, palette=box_colors, ax=ax, linewidth=1.5)

    stats = df.groupby('status')['NDVI_smooth'].agg(['mean', 'std'])
    for i, s in enumerate(box_order):
        if s in stats.index:
            ax.text(i, stats.loc[s, 'mean'] + 0.05,
                    f"μ={stats.loc[s, 'mean']:.2f}\nσ={stats.loc[s, 'std']:.2f}",
                    ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_title('Distribusi NDVI per Kelompok', fontsize=14, fontweight='bold')
    ax.set_xlabel('Kelompok')
    ax.set_ylabel('NDVI Smoothed')
    ax.set_xticklabels(box_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig

def plot_spatial_map(df, valid_statuses, NOISE_COLOR, UNSELECTED_COLOR, VALID_PALETTE):
    nr = df['grid_row'].max() + 1
    nc = df['grid_col'].max() + 1

    status_int_map = {'unselected': 0, 'noise': 1}
    for i, s in enumerate(valid_statuses):
        status_int_map[s] = i + 2

    grid_matrix_status = np.full((nr, nc), np.nan)
    gmap = df[['grid_row', 'grid_col', 'status']].drop_duplicates(subset=['grid_row', 'grid_col'])
    grid_matrix_status[gmap['grid_row'].values, gmap['grid_col'].values] = gmap['status'].map(status_int_map).values

    n_valid = len(valid_statuses)
    all_colors = [UNSELECTED_COLOR, NOISE_COLOR] + [plt.matplotlib.colors.rgb2hex(c) for c in VALID_PALETTE[:n_valid]]
    cmap_cluster = ListedColormap(all_colors)
    masked_grid = np.ma.masked_invalid(grid_matrix_status)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.pcolormesh(np.arange(nc + 1), np.arange(nr + 1), masked_grid,
                  cmap=cmap_cluster, edgecolors='white', linewidth=0.3,
                  vmin=0, vmax=len(all_colors))
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title('Peta Spasial Distribusi Cluster', fontsize=14, fontweight='bold')
    ax.set_xlabel('Kolom Grid (Longitude)')
    ax.set_ylabel('Baris Grid (Latitude)')
    ax.set_xticks([])
    ax.set_yticks([])

    legend_elements = [
        Patch(facecolor='white', edgecolor='gray', label='Tidak Ada Data'),
        Patch(facecolor=UNSELECTED_COLOR, edgecolor='white', label='Unselected'),
        Patch(facecolor=NOISE_COLOR, edgecolor='white', label='Noise'),
    ] + [Patch(facecolor=plt.matplotlib.colors.rgb2hex(VALID_PALETTE[i % len(VALID_PALETTE)]),
               edgecolor='white', label=f'Cluster {i}') for i in range(n_valid)]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, frameon=True)
    fig.tight_layout()
    return fig