# save as step1_clean_parse.py and run: python step1_clean_parse.py
import pandas as pd
import numpy as np
import re
from pathlib import Path

DATA_DIR = Path('data')  # update if your files are elsewhere
CROP_FILE = DATA_DIR / 'crop_conditions.csv'   # dataset 1
REGION_FILE = DATA_DIR / 'region_conditions.csv'  # dataset 2

def safe_read(csv_path):
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f" -> shape: {df.shape}")
    return df

def parse_range_to_stats(s):
    """
    Parse text ranges like:
      - "20-30"
      - "20 - 30 °C"
      - "20–30" (en dash)
      - "6.0-7.5"
      - "opt 20-25" or "18 to 25"
    Returns (min, mean, max) as floats or (np.nan, np.nan, np.nan) on failure.
    """
    if pd.isna(s):
        return (np.nan, np.nan, np.nan)
    if isinstance(s, (int, float)):
        return (float(s), float(s), float(s))
    text = str(s).strip()
    # find two numbers in string
    nums = re.findall(r'[-+]?\d*\.\d+|\d+', text)
    if len(nums) == 0:
        return (np.nan, np.nan, np.nan)
    if len(nums) == 1:
        v = float(nums[0])
        return (v, v, v)
    # take first two as bounds (if more, just take first two)
    a = float(nums[0])
    b = float(nums[1])
    lo, hi = (min(a, b), max(a, b))
    mean = (lo + hi) / 2.0
    return (lo, mean, hi)

def clean_crop_df(df):
    df = df.copy()
    # normalize column names: lower, strip, replace spaces and special chars
    df.columns = [re.sub(r'[^0-9a-zA-Z_]+', '_', c.strip().lower()) for c in df.columns]
    # expected col name guesses (adapt if your actual names differ)
    # examples: crop, yield_qtl_per_acre, price_per_qtl_rs, main_soiltype, sub_soiltype, n_kg_ha, p₂o₅_kg_ha, k₂o_kg_ha, temperature__c__optimal_range_, humidity__typical_, ph__range_
    # try to find columns by substring
    def find(col_sub):
        for c in df.columns:
            if col_sub in c:
                return c
        return None

    # map commonly-named cols
    col_crop = find('crop') or 'crop'
    col_n = find('n_') or find('n_kg') or find('n_kg_ha') or find('n') 
    col_p = find('p') or find('p_2o5') or find('p_') 
    col_k = find('k') or find('k_2o') or find('k_')
    col_temp = find('temp') or find('temperature')
    col_humidity = find('humidity')
    col_ph = find('ph')
    col_mainsoil = find('main_soil') or find('main_soiltype') or find('soiltype')

    print("Detected columns (crop df):")
    print(" crop:", col_crop)
    print(" N:", col_n, " P:", col_p, " K:", col_k)
    print(" temp:", col_temp, " humidity:", col_humidity, " ph:", col_ph)
    print(" main soil:", col_mainsoil)

    # coerce numeric NPK
    for col_hint, target in [(col_n, 'N'), (col_p, 'P'), (col_k, 'K')]:
        if col_hint and col_hint in df.columns:
            df[target] = pd.to_numeric(df[col_hint].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
        else:
            df[target] = np.nan

    # parse temperature and ph ranges
    df[['temp_lo', 'temp_mean', 'temp_hi']] = df.get(col_temp).apply(lambda x: pd.Series(parse_range_to_stats(x)))
    df[['ph_lo', 'ph_mean', 'ph_hi']] = df.get(col_ph).apply(lambda x: pd.Series(parse_range_to_stats(x)))
    # humidity may be percentage or range
    df[['humidity_lo', 'humidity_mean', 'humidity_hi']] = df.get(col_humidity).apply(lambda x: pd.Series(parse_range_to_stats(x)))

    # normalize soil type column
    if col_mainsoil and col_mainsoil in df.columns:
        df['main_soiltype'] = df[col_mainsoil].astype(str).str.lower().str.strip()
    else:
        df['main_soiltype'] = df.get(col_mainsoil, '').astype(str).str.lower().str.strip()

    # ensure crop name normalized
    df['crop'] = df[col_crop].astype(str).str.strip().str.lower()

    # keep important columns
    keep = ['crop', 'yield_qtl_per_acre', 'price_per_qtl_rs', 'main_soiltype',
            'N', 'P', 'K',
            'temp_lo','temp_mean','temp_hi',
            'humidity_lo','humidity_mean','humidity_hi',
            'ph_lo','ph_mean','ph_hi']
    existing = [c for c in keep if c in df.columns or c in ['crop','main_soiltype','N','P','K','temp_lo','temp_mean','temp_hi','humidity_lo','humidity_mean','humidity_hi','ph_lo','ph_mean','ph_hi']]
    return df[existing]

def clean_region_df(df):
    df = df.copy()
    df.columns = [re.sub(r'[^0-9a-zA-Z_]+', '_', c.strip().lower()) for c in df.columns]
    # expected: dist, taluq, crop, season, avg_temp_rabi, avg_temp_kharif, avg_rainfall
    print("Detected columns (region df):", df.columns.tolist())
    # normalize text
    for c in ['dist','district','taluq','taluk']:
        if c in df.columns:
            df['district'] = df[c].astype(str).str.lower().str.strip()
            break
    for c in ['taluq','taluk','taluka']:
        if c in df.columns:
            df['taluq'] = df[c].astype(str).str.lower().str.strip()
            break
    if 'crop' in df.columns:
        df['crop'] = df['crop'].astype(str).str.lower().str.strip()
    # rainfall numeric
    rain_col = None
    for c in df.columns:
        if 'rain' in c:
            rain_col = c
            break
    if rain_col:
        df['avg_rainfall'] = pd.to_numeric(df[rain_col].astype(str).str.replace(r'[^\d\.]', '', regex=True), errors='coerce')
    # temperatures for seasons
    for season in ['rabi','kharif','kharif_rabi','kharifrabi']:
        cand = [c for c in df.columns if season in c]
        if cand:
            df[f'avg_temp_{season}'] = pd.to_numeric(df[cand[0]].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
    # fallback: any column with 'temp' map to avg_temp
    temp_cols = [c for c in df.columns if 'temp' in c and 'avg_temp' not in c]
    for i,c in enumerate(temp_cols):
        df[f'avg_temp_custom_{i}'] = pd.to_numeric(df[c].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
    # ensure district/taluq present
    if 'district' not in df.columns:
        df['district'] = df.iloc[:,0].astype(str).str.lower().str.strip()
    if 'taluq' not in df.columns:
        df['taluq'] = df.iloc[:,1].astype(str).str.lower().str.strip()
    # season column
    if 'season' in df.columns:
        df['season'] = df['season'].astype(str).str.lower().str.strip()
    else:
        df['season'] = np.nan

    # keep minimal useful columns
    keep = ['district','taluq','crop','season','avg_rainfall']
    existing = [c for c in keep if c in df.columns]
    # also include any avg_temp_* columns
    temp_cols = [c for c in df.columns if c.startswith('avg_temp_')]
    existing += temp_cols
    return df[existing]

def main():
    crop_df = safe_read(CROP_FILE)
    region_df = safe_read(REGION_FILE)

    crop_clean = clean_crop_df(crop_df)
    region_clean = clean_region_df(region_df)

    print("\nCROP dataframe after cleaning sample:")
    print(crop_clean.head(6).T)

    print("\nREGION dataframe after cleaning sample:")
    print(region_clean.head(6).T)

    # report missingness
    print("\nMissing value counts (crop):")
    print(crop_clean.isna().sum())
    print("\nMissing value counts (region):")
    print(region_clean.isna().sum())

    # basic stats
    print("\nCrop temp_mean stats:")
    print(crop_clean['temp_mean'].describe())

    print("\nRegion avg_rainfall stats:")
    if 'avg_rainfall' in region_clean.columns:
        print(region_clean['avg_rainfall'].describe())
    else:
        print("avg_rainfall not found in region dataset columns.")

    # save cleaned versions for next step
    OUT_DIR = DATA_DIR / 'cleaned'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    crop_clean.to_csv(OUT_DIR / 'crop_cleaned.csv', index=False)
    region_clean.to_csv(OUT_DIR / 'region_cleaned.csv', index=False)
    print(f"\nSaved cleaned CSVs to {OUT_DIR}")

if __name__ == '__main__':
    main()
