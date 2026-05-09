import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import random

def load_and_preprocess_data(csv_2023, csv_2024, decimals=5):
    df_2023 = pd.read_csv(csv_2023, parse_dates=['tanggal'])
    df_2024 = pd.read_csv(csv_2024, parse_dates=['tanggal'])

    df_2023['tahun'] = '2023'
    df_2024['tahun'] = '2024'
    df_combined = pd.concat([df_2023, df_2024], ignore_index=True)

    df_clean = (
        df_combined
        .drop_duplicates(subset=['lat_y', 'lon_x', 'tanggal'], keep='first')
        .sort_values(['lat_y', 'lon_x', 'tanggal'])
        .reset_index(drop=True)
    )
    df_clean['id_lokasi'] = 'LOC_' + df_clean.groupby(['lat_y', 'lon_x']).ngroup().astype(str).str.zfill(4)
    df_clean['cluster_id'] = -1

    # Grid Mapping
    df_clean['lat_grid'] = df_clean['lat_y'].round(decimals)
    df_clean['lon_grid'] = df_clean['lon_x'].round(decimals)
    
    unique_lats = np.sort(df_clean['lat_grid'].unique())[::-1]
    unique_lons = np.sort(df_clean['lon_grid'].unique())
    lat_map = {v: i for i, v in enumerate(unique_lats)}
    lon_map = {v: i for i, v in enumerate(unique_lons)}
    n_cols = len(unique_lons)

    df_clean['grid_row'] = df_clean['lat_grid'].map(lat_map).astype('int16')
    df_clean['grid_col'] = df_clean['lon_grid'].map(lon_map).astype('int16')
    df_clean['grid_id'] = (df_clean['grid_row'] * n_cols + df_clean['grid_col'] + 1).astype('int32')
    df_clean.drop(columns=['lat_grid', 'lon_grid'], inplace=True)

    return df_clean.sort_values(['id_lokasi', 'tanggal']).reset_index(drop=True)

def smooth_and_resample(df_clean, window_size=31, poly_order=2):
    # 1. Ekstrak metadata statis per lokasi agar aman dari KeyError Pandas
    loc_meta = df_clean.groupby('id_lokasi')[['lat_y', 'lon_x', 'grid_row', 'grid_col', 'grid_id']].first()
    
    results = []
    # 2. Proses per lokasi menggunakan loop (lebih stabil & sering lebih cepat di Pandas >2.0)
    for loc_id, group in df_clean.groupby('id_lokasi'):
        series = group.set_index('tanggal')['NDVI']
        daily = series.resample('D').interpolate(method='linear').reset_index()
        
        if len(daily) >= window_size:
            daily['NDVI_smooth'] = savgol_filter(
                daily['NDVI'].values, window_length=window_size, polyorder=poly_order
            ).astype('float32')
        else:
            daily['NDVI_smooth'] = daily['NDVI'].astype('float32')
            
        daily['id_lokasi'] = loc_id
        daily['lat_y'] = loc_meta.loc[loc_id, 'lat_y']
        daily['lon_x'] = loc_meta.loc[loc_id, 'lon_x']
        daily['grid_row'] = loc_meta.loc[loc_id, 'grid_row']
        daily['grid_col'] = loc_meta.loc[loc_id, 'grid_col']
        daily['grid_id'] = loc_meta.loc[loc_id, 'grid_id']
        
        results.append(daily)
        
    return pd.concat(results, ignore_index=True)

def filter_and_sample_data(df_smooth, target_year=2024, n_sample=500, seed=42):
    df_year = df_smooth.loc[df_smooth['tanggal'].dt.year == target_year].copy()
    all_locs = df_year['id_lokasi'].unique()
    random.seed(seed)
    sample_locs = random.sample(list(all_locs), k=min(n_sample, len(all_locs)))
    df_sample = df_year[df_year['id_lokasi'].isin(sample_locs)].copy()
    return df_year, df_sample, sample_locs
