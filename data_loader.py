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
    def process_group(group):
        series = group.set_index('tanggal')['NDVI']
        daily = series.resample('D').interpolate(method='linear').reset_index()
        if len(daily) >= window_size:
            daily['NDVI_smooth'] = savgol_filter(
                daily['NDVI'].values, window_length=window_size, polyorder=poly_order
            ).astype('float32')
        else:
            daily['NDVI_smooth'] = daily['NDVI'].astype('float32')
            
        daily['id_lokasi'] = group['id_lokasi'].iloc[0]
        daily['lat_y'] = group['lat_y'].iloc[0]
        daily['lon_x'] = group['lon_x'].iloc[0]
        daily['grid_row'] = group['grid_row'].iloc[0]
        daily['grid_col'] = group['grid_col'].iloc[0]
        daily['grid_id'] = group['grid_id'].iloc[0]
        return daily

    processed = df_clean.groupby('id_lokasi').apply(process_group).reset_index(drop=True)
    for col in ['NDVI', 'lat_y', 'lon_x']:
        processed[col] = processed[col].astype('float32')
    processed['id_lokasi'] = processed['id_lokasi'].astype('category')
    return processed

def filter_and_sample_data(df_smooth, target_year=2024, n_sample=500, seed=42):
    df_year = df_smooth.loc[df_smooth['tanggal'].dt.year == target_year].copy()
    all_locs = df_year['id_lokasi'].unique()
    random.seed(seed)
    sample_locs = random.sample(list(all_locs), k=min(n_sample, len(all_locs)))
    df_sample = df_year[df_year['id_lokasi'].isin(sample_locs)].copy()
    return df_year, df_sample, sample_locs