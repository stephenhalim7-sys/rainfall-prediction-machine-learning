import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Membaca Data
INPUT_FILES = [
    "sinoptik 2022-01-01 to 2022-12-31.csv",
    "sinoptik 2023-01-01 to 2023-12-31.csv",
    "sinoptik 2024-01-01 to 2024-12-31.csv",
    "sinoptik 2025-01-01 to 2025-12-31.csv",
    "sinoptik 2026-01-01 to 2026-01-31.csv",
]
INPUT_DIR   = "PKL"
OUTPUT_DIR  = r"C:\Users\Stephen\Documents\VSCode\PKL"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "data_harian_bmkg_2022_2026.csv")

INVALID_FLAGS = [9999, -9999, 999.0, 9999.0, -9999.0]

VALID_RANGE = {
    "kecepatan_angin_knot"   : (0, 60),
    "tekanan_udara_qff_hpa"  : (980, 1030),
    "suhu_max_c"             : (20, 45),
    "suhu_min_c"             : (15, 35),
    "kelembapan_relatif_pct" : (0, 100),
    "curah_hujan_mm"         : (0, 300),
}

KOLOM_TARGET = {
    "timestamp"               : "tanggal_jam",
    "wind_speed_ff"           : "kecepatan_angin_knot",
    "pressure_qff_mb_derived" : "tekanan_udara_qff_hpa",
    "temp_max_c_txtxtx"       : "suhu_max_c",
    "temp_min_c_tntntn"       : "suhu_min_c",
    "relative_humidity_pc"    : "kelembapan_relatif_pct",
    "rainfall_24h_rrrr"       : "curah_hujan_mm",
}

def read_raw_file(filepath: str) -> pd.DataFrame:
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
    if first_line.startswith("Form Sinop"):
        df = pd.read_csv(filepath, skiprows=1, on_bad_lines='skip')
    else:
        df = pd.read_csv(filepath, on_bad_lines='skip')
    return df

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = {}
    for orig, new in KOLOM_TARGET.items():
        result[new] = df[orig].copy() if orig in df.columns else np.nan
    return pd.DataFrame(result)

def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['tanggal_jam'] = pd.to_datetime(
        df['tanggal_jam'].astype(str).str.strip() + ":00:00",
        format="%Y-%m-%d %H:%M:%S",
        errors='coerce'
    )
    n_invalid = df['tanggal_jam'].isna().sum()
    if n_invalid > 0:
        print(f"      ⚠ {n_invalid} baris timestamp tidak valid → dihapus")
        df = df.dropna(subset=['tanggal_jam'])
    df['tanggal'] = df['tanggal_jam'].dt.date
    df['jam']     = df['tanggal_jam'].dt.hour
    return df

# Membersihkan Data
def clean_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'curah_hujan_mm' in df.columns:
        df['curah_hujan_mm'] = df['curah_hujan_mm'].replace('-', np.nan)
        df['curah_hujan_mm'] = pd.to_numeric(df['curah_hujan_mm'], errors='coerce')
    for col in df.columns:
        if col in ['tanggal_jam', 'tanggal']:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        mask = df[col].isin(INVALID_FLAGS)
        if mask.sum() > 0:
            df.loc[mask, col] = np.nan
    for col, (lo, hi) in VALID_RANGE.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            if mask.sum() > 0:
                df.loc[mask, col] = np.nan
    return df

# Agregasi Data
def aggregate_to_daily(df):
    COLS_MEAN = [
        'kelembapan_relatif_pct',
        'kecepatan_angin_knot',
        'tekanan_udara_qff_hpa'
    ]

    df_00 = df[df['jam'] == 0].copy()

    # Ambil Tmax dan Tmin per hari
    tmax_harian = df.groupby('tanggal')['suhu_max_c'].max()
    tmin_harian = df.groupby('tanggal')['suhu_min_c'].min()

    # Hitung Tavg = (Tmax + Tmin) / 2, simpan langsung sebagai suhu_rata2_c
    tavg_harian = ((tmax_harian + tmin_harian) / 2).rename('suhu_rata2_c')

    # Curah hujan digeser 1 hari ke belakang
    hujan_harian = df_00[['tanggal', 'curah_hujan_mm']].copy()
    hujan_harian['tanggal'] = (
        pd.to_datetime(hujan_harian['tanggal']) - pd.Timedelta(days=1)
    )
    hujan_harian = hujan_harian.set_index('tanggal')

    # Rata-rata parameter per jam
    df_mean = df.groupby('tanggal')[COLS_MEAN].mean()

    # Gabung: Tavg + parameter rata-rata + curah hujan
    daily = tavg_harian.to_frame().join(df_mean, how='outer')
    daily = daily.join(hujan_harian, how='outer')

    daily.index = pd.to_datetime(daily.index)
    daily = (
        daily.reset_index()
             .rename(columns={'index': 'tanggal'})
             .sort_values('tanggal')
             .reset_index(drop=True)
    )

    # Buang tanggal sebelum 2022-01-01
    daily = daily[daily['tanggal'] >= '2022-01-01']
    return daily

# Menghapus Duplikat
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates(subset=['tanggal'], keep='first')
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"    ⚠ {n_removed} baris duplikat dihapus")
    else:
        print(f"    ✓ Tidak ada duplikat tanggal")
    return df

# Menghapus Missing Values
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values('tanggal').reset_index(drop=True)

    # Flag missing berdasarkan kolom output final
    penting = [c for c in [
        'curah_hujan_mm',
        'suhu_rata2_c',
        'kelembapan_relatif_pct',
        'kecepatan_angin_knot',
        'tekanan_udara_qff_hpa'
    ] if c in df.columns]
    df['flag_missing'] = df[penting].isnull().any(axis=1).astype(int)

    n_missing_days = df['flag_missing'].sum()
    print(f"    Hari dengan ≥1 missing value: {n_missing_days} hari ({n_missing_days/len(df)*100:.1f}%)")

    # Curah hujan: NaN → 0
    if 'curah_hujan_mm' in df.columns:
        n = df['curah_hujan_mm'].isna().sum()
        df['curah_hujan_mm'] = df['curah_hujan_mm'].fillna(0)
        if n > 0:
            print(f"    curah_hujan_mm : {n} NaN → diisi 0")

    # Kolom lain: interpolasi linear
    num_cols = [c for c in df.select_dtypes(include='number').columns
                if c not in ['flag_missing', 'curah_hujan_mm']]
    for col in num_cols:
        n = df[col].isna().sum()
        if n > 0:
            df[col] = (df[col]
                       .interpolate(method='linear', limit_direction='both')
                       .ffill()
                       .bfill())
            sisa = df[col].isna().sum()
            print(f"    {col:<35}: {n} NaN → interpolasi → {sisa} sisa")

    return df

# Validasi
def validate_and_report(df: pd.DataFrame):
    print(f"\n{'='*65}")
    print("  LAPORAN VALIDASI DATA AKHIR")
    print(f"{'='*65}")
    print(f"  Total baris     : {len(df):,} hari")
    print(f"  Rentang tanggal : {df['tanggal'].min()} s/d {df['tanggal'].max()}")
    print(f"  Total kolom     : {len(df.columns)}")
    print(f"  Kolom           : {list(df.columns)}\n")

    print(f"  Missing values per kolom:")
    for col, n in df.isnull().sum().items():
        status = "✓" if n == 0 else "✗"
        print(f"    {status}  {col:<35}: {n}")

    tanggal_range = pd.date_range(df['tanggal'].min(), df['tanggal'].max(), freq='D')
    missing_dates = set(tanggal_range.date) - set(df['tanggal'].dt.date)
    if missing_dates:
        print(f"\n  ⚠ {len(missing_dates)} hari tidak ada data:")
        for d in sorted(missing_dates)[:5]:
            print(f"     {d}")
        if len(missing_dates) > 5:
            print(f"     ... dan {len(missing_dates)-5} lainnya")
    else:
        print(f"\n  ✓ Data lengkap, tidak ada hari yang hilang")

    print(f"\n  Hari dengan flag_missing=1 : {df['flag_missing'].sum()}")
    print(f"\n  Statistik deskriptif:")
    print(df.drop(columns=['tanggal', 'flag_missing']).describe().round(2).to_string())
    print(f"{'='*65}")

# Main
def main():
    print(f"\n{'='*65}")
    print("  PREPROCESSING DATA SINOPTIK BMKG TANJUNG PERAK SURABAYA")
    print(f"{'='*65}\n")

    print("STEP 1 — Membaca file mentah")
    frames = []
    for fname in INPUT_FILES:
        fpath = os.path.join(INPUT_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  ⚠ File tidak ditemukan: {fpath} — dilewati")
            continue
        df = read_raw_file(fpath)
        df = standardize_columns(df)
        df = parse_timestamp(df)
        df = clean_invalid_values(df)
        frames.append(df)
        print(f"  ✓ {fname}: {len(df):,} baris per-jam")

    if not frames:
        raise FileNotFoundError("Tidak ada file yang berhasil dibaca!")

    print(f"\nSTEP 5 — Menggabungkan & agregasi ke data harian")
    df_all   = pd.concat(frames, ignore_index=True)
    df_daily = aggregate_to_daily(df_all)
    print(f"  Total jam (semua file) : {len(df_all):,}")
    print(f"  Total hari setelah agregasi: {len(df_daily):,}")

    print(f"\nSTEP 6 — Hapus duplikat tanggal")
    df_daily = remove_duplicates(df_daily)

    print(f"\nSTEP 7 — Tangani missing values")
    df_daily = handle_missing_values(df_daily)

    validate_and_report(df_daily)

    df_daily['tanggal'] = df_daily['tanggal'].astype(str)
    df_daily = df_daily.round(2)
    df_daily.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"\n✅ SELESAI! File tersimpan: {OUTPUT_FILE}")
    print(f"   {len(df_daily):,} baris × {len(df_daily.columns)} kolom\n")
    print(f"   Kolom output: {list(df_daily.columns)}")


if __name__ == "__main__":
    main()