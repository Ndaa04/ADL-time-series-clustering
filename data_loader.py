import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# Nama file CSV (harus ada di folder yang sama dengan app.py)
PATH_2023 = "Data_NDVI_Lamongan_2023.csv"
PATH_2024 = "Data_NDVI_Lamongan_2024.csv"


def load_raw_data(progress_callback=None):
    """Baca CSV lokal, gabungkan, bersihkan, buat grid spasial."""
    if progress_callback: progress_callback(0.10, "📂 Membaca data 2023...")
    df_2023 = pd.read_csv(PATH_2023, parse_dates=['tanggal'])
    df_2023['tahun'] = '2023'

    if progress_callback: progress_callback(0.30, "📂 Membaca data 2024...")
    df_2024 = pd.read_csv(PATH_2024, parse_dates=['tanggal'])
    df_2024['tahun'] = '2024'

    if progress_callback: progress_callback(0.50, "🔗 Menggabungkan & membersihkan...")
    df_combined = pd.concat([df_2023, df_2024], ignore_index=True)
    df_clean = (
        df_combined
        .drop_duplicates(subset=['lat_y', 'lon_x', 'tanggal'], keep='first')
        .sort_values(['lat_y', 'lon_x', 'tanggal'])
        .reset_index(drop=True)
        .copy()
    )
    df_clean['id_lokasi'] = (
        'LOC_' + df_clean.groupby(['lat_y', 'lon_x']).ngroup().astype(str).str.zfill(4)
    )
    df_clean['cluster_id'] = -1

    if progress_callback: progress_callback(0.70, "🗺️ Membuat grid spasial...")
    DECIMALS = 5
    df_clean['lat_grid'] = df_clean['lat_y'].round(DECIMALS)
    df_clean['lon_grid'] = df_clean['lon_x'].round(DECIMALS)

    unique_lats = np.sort(df_clean['lat_grid'].unique())[::-1]
    unique_lons = np.sort(df_clean['lon_grid'].unique())
    lat_map = {v: i for i, v in enumerate(unique_lats)}
    lon_map = {v: i for i, v in enumerate(unique_lons)}

    df_clean['grid_row'] = df_clean['lat_grid'].map(lat_map).astype('int16')
    df_clean['grid_col'] = df_clean['lon_grid'].map(lon_map).astype('int16')
    n_rows, n_cols = len(unique_lats), len(unique_lons)
    df_clean.drop(columns=['lat_grid', 'lon_grid'], inplace=True)

    if progress_callback: progress_callback(1.0, "✅ Data siap!")
    return df_clean, n_rows, n_cols


def apply_smoothing(df_raw, window_size=31, poly_order=2, progress_callback=None):
    """Resample harian + Savitzky-Golay smoothing per lokasi."""
    groups = list(df_raw.groupby('id_lokasi'))
    total = len(groups)
    results = []

    for idx, (loc, group) in enumerate(groups):
        series = group.set_index('tanggal')['NDVI']
        daily = series.resample('D').interpolate(method='linear').reset_index()

        if len(daily) >= window_size:
            daily['NDVI_smooth'] = savgol_filter(
                daily['NDVI'].values, window_length=window_size, polyorder=poly_order
            ).astype('float32')
        else:
            daily['NDVI_smooth'] = daily['NDVI'].astype('float32')

        meta = group[['id_lokasi', 'lat_y', 'lon_x', 'tahun', 'grid_row', 'grid_col']].iloc[0]
        for col in ['id_lokasi', 'lat_y', 'lon_x', 'tahun', 'grid_row', 'grid_col']:
            daily[col] = meta[col]

        results.append(daily[['id_lokasi', 'tanggal', 'NDVI', 'NDVI_smooth',
                               'lat_y', 'lon_x', 'tahun', 'grid_row', 'grid_col']])

        if progress_callback and idx % 50 == 0:
            pct = 0.05 + 0.90 * (idx / total)
            progress_callback(pct, f"🔬 Smoothing {idx+1}/{total} lokasi...")

    df_final = pd.concat(results, ignore_index=True)
    if progress_callback: progress_callback(1.0, "✅ Smoothing selesai!")
    return df_final
