import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

plt.rcParams.update({
    'figure.dpi': 120, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'font.family': 'DejaVu Sans'
})
sns.set_palette("husl")

print("=" * 65)
print("  PRAKTIK KERJA LAPANGAN (PKL)")
print("  PREDIKSI CURAH HUJAN – BMKG MARITIM TANJUNG PERAK SURABAYA")
print("  Linear Regression vs Random Forest Regression ")
print("  Stephen Lionel Halim           (181231042)")
print("  Muhammad Fazle Mawla Yusup     (181231074)")
print("=" * 65)

# Membaca File
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, 'data_harian_bmkg_2022_2026.csv')

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        f"File tidak ditemukan: {FILE_PATH}\n"
        "Pastikan 'data_harian_bmkg_2022_2026.csv' berada di folder yang sama."
    )

print(f"\n    📂 File ditemukan: {os.path.basename(FILE_PATH)}")
print("\n[1] Membaca Dataset ...")

df = pd.read_csv(FILE_PATH, parse_dates=['tanggal'])

df.rename(columns={
    'tanggal'                  : 'Tanggal',
    'suhu_rata2_c'             : 'Tavg',
    'kelembapan_relatif_pct'   : 'RH_avg',
    'kecepatan_angin_knot'     : 'ff_avg',
    'tekanan_udara_qff_hpa'    : 'P0',
    'curah_hujan_mm'           : 'RR',
    'flag_missing'             : 'flag_missing',
}, inplace=True)

df.sort_values('Tanggal', inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"    ✓ Total baris dibaca   : {len(df)} hari")
print(f"    ✓ Rentang waktu        : {df['Tanggal'].min().date()} s.d. {df['Tanggal'].max().date()}")
print(f"    ✓ Kolom tersedia       : {list(df.columns)}")

print("\n[2] Membuat Fitur Temporal (Lag, Rolling, Musiman) ...")

BASE_COLS = ['Tavg', 'RH_avg', 'ff_avg', 'P0']

for col in ['RR', 'RH_avg', 'P0']:
    for lag in [1, 2, 3]:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

for col in BASE_COLS + ['RR']:
    df[f'{col}_roll3'] = df[col].shift(1).rolling(window=3, min_periods=3).mean()
    df[f'{col}_roll7'] = df[col].shift(1).rolling(window=7, min_periods=7).mean()

df['P0_delta'] = df['P0'] - df['P0'].shift(1)

df['month'] = df['Tanggal'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

n_before = len(df)
df.dropna(subset=[c for c in df.columns if 'lag' in c or 'roll' in c or 'delta' in c],
           inplace=True)
df.reset_index(drop=True, inplace=True)
n_after = len(df)

print(f"    ✓ Fitur lag dibuat     : RR, RH_avg, P0 (lag 1–3 hari)")
print(f"    ✓ Fitur rolling dibuat : rata-rata 3 & 7 hari (Tavg, RH_avg, ff_avg, P0, RR)")
print(f"    ✓ Fitur musiman        : bulan (sin/cos), delta tekanan udara")
print(f"    ✓ Baris dibuang (histori awal belum lengkap): {n_before - n_after} hari")
print(f"    ✓ Total baris siap dipakai untuk modeling   : {n_after} hari")

FEATURE_COLS = (
    BASE_COLS
    + [f'{c}_lag{l}' for c in ['RR', 'RH_avg', 'P0'] for l in [1, 2, 3]]
    + [f'{c}_roll3' for c in BASE_COLS + ['RR']]
    + [f'{c}_roll7' for c in BASE_COLS + ['RR']]
    + ['P0_delta', 'month_sin', 'month_cos']
)

LABEL_MAP = {
    'Tavg'   : 'Suhu Rata-rata (C)',
    'RH_avg' : 'Kelembapan Udara (%)',
    'ff_avg' : 'Kecepatan Angin (knot)',
    'P0'     : 'Tekanan Udara (hPa)',
    'RR'     : 'Curah Hujan (mm)',
}
for c in FEATURE_COLS:
    if c not in LABEL_MAP:
        LABEL_MAP[c] = c

# Statistik Deskriptif
print("\n[3] Statistik Deskriptif (variabel dasar) ...")
desc = df[BASE_COLS + ['RR']].describe().T
DESC_LABEL = {
    'Tavg'   : 'Suhu Udara Rata-rata',
    'RH_avg' : 'Kelembapan Udara',
    'ff_avg' : 'Kecepatan Angin',
    'P0'     : 'Tekanan Udara',
    'RR'     : 'Curah Hujan',
}
desc.rename(index=DESC_LABEL, inplace=True)
desc['CV (%)'] = (desc['std'] / desc['mean'].abs() * 100).round(2)
print(desc.round(3).to_string())

# Matriks Korelasi
print("\n[4] Membuat Matriks Korelasi ...")
corr_df = df[BASE_COLS + ["RR"]].corr()
corr_df.rename(columns=LABEL_MAP, index=LABEL_MAP, inplace=True)

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1, linewidths=.4,
            ax=ax, cbar_kws={'shrink': .8})
ax.set_title('Matriks Korelasi Antar Variabel Dasar\n'
             '(BMKG Tanjung Perak Surabaya)', fontsize=13, fontweight='bold', pad=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('korelasi_variabel.png', bbox_inches='tight')
plt.show()
print("    ✓ Disimpan → korelasi_variabel.png")

# Split Berbasis Waktu
print("\n[5] Membagi Data Berdasarkan Urutan Waktu (time-based split) ...")
print("    ⚠ Tidak memakai train_test_split acak karena ini data deret waktu.")
print("      Split acak akan membuat model 'mengintip' pola masa depan (data leakage).")

X = df[FEATURE_COLS].copy()
y = df['RR'].copy()
tanggal = df['Tanggal'].copy()

split_idx = int(len(df) * 0.80)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
tgl_train, tgl_test = tanggal.iloc[:split_idx], tanggal.iloc[split_idx:]

print(f"    • Training : {len(X_train)} sampel  ({tgl_train.min().date()} s.d. {tgl_train.max().date()})")
print(f"    • Testing  : {len(X_test)} sampel  ({tgl_test.min().date()} s.d. {tgl_test.max().date()})")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Melatih Model
print("\n[6] Melatih Model ...")

lr_model = LinearRegression()
lr_model.fit(X_train_sc, y_train)
print("    ✓ Linear Regression selesai")

print("    ⏳ Tuning hyperparameter Random Forest (RandomizedSearchCV + TimeSeriesSplit) ...")
print("       (proses ini bisa makan waktu beberapa puluh detik – mencoba banyak kombinasi)")

param_dist = {
    'n_estimators'      : [100, 200, 300, 500],
    'max_depth'         : [3, 5, 7, 9, 12, None],
    'min_samples_split' : [2, 5, 10, 15, 20],
    'min_samples_leaf'  : [1, 2, 4, 8, 12],
    'max_features'      : ['sqrt', 'log2', 0.5, 0.7, None],
}

tscv = TimeSeriesSplit(n_splits=5)

search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=40,
    scoring='neg_root_mean_squared_error',
    cv=tscv,
    random_state=42,
    n_jobs=-1,
    verbose=0,
)
search.fit(X_train, y_train)

rf_model = search.best_estimator_
print("    ✓ Random Forest (tuned) selesai")
print(f"    ✓ Parameter terbaik ditemukan:")
for k, v in search.best_params_.items():
    print(f"       • {k:<18}: {v}")
print(f"    ✓ Skor cross-validation terbaik (RMSE) : {-search.best_score_:.4f}")

# Prediksi dan Evaluasi Model
print("\n[7] Evaluasi Model ...")

def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {'Model': name, 'RMSE': rmse, 'MAE': mae, 'R²': r2}

y_pred_lr = np.clip(lr_model.predict(X_test_sc), 0, None)
y_pred_rf = np.clip(rf_model.predict(X_test),    0, None)

metrics_lr = evaluate('Linear Regression', y_test, y_pred_lr)
metrics_rf = evaluate('Random Forest',     y_test, y_pred_rf)

results = pd.DataFrame([metrics_lr, metrics_rf]).set_index('Model')
disp = results.copy()
for col in ['RMSE', 'MAE', 'R²']:
    disp[col] = disp[col].map('{:.4f}'.format)

print("\n    ┌──────────────────────────────────────────────────────────┐")
print("    │         TABEL PERBANDINGAN METRIK EVALUASI MODEL         │")
print("    └──────────────────────────────────────────────────────────┘")
print(disp.to_string())

y_pred_rf_train = np.clip(rf_model.predict(X_train), 0, None)
rmse_rf_train   = np.sqrt(mean_squared_error(y_train, y_pred_rf_train))
print(f"\n    🔍 Cek overfitting Random Forest:")
print(f"       • RMSE Training : {rmse_rf_train:.4f}")
print(f"       • RMSE Testing  : {metrics_rf['RMSE']:.4f}")
gap = (metrics_rf['RMSE'] - rmse_rf_train) / rmse_rf_train * 100
print(f"       • Selisih       : {gap:.1f}% (semakin besar -> semakin overfit)")

ambang = 1.0
aktual_hujan = (y_test.values > ambang).astype(int)
pred_lr_hujan = (y_pred_lr > ambang).astype(int)
pred_rf_hujan = (y_pred_rf > ambang).astype(int)
akurasi_lr = (aktual_hujan == pred_lr_hujan).mean() * 100
akurasi_rf = (aktual_hujan == pred_rf_hujan).mean() * 100
print(f"\n    🌧  Akurasi klasifikasi 'hujan vs tidak hujan' (ambang {ambang} mm):")
print(f"       • Linear Regression : {akurasi_lr:.2f}%")
print(f"       • Random Forest     : {akurasi_rf:.2f}%")

# Menyimpan Hasil Prediksi Harian ke CSV (khusus Data Uji vs Data Aktual)
print("\n[7b] Menyimpan Hasil Prediksi Harian — Data Uji vs Data Aktual ke CSV ...")
print(f"    ℹ Baris CSV ini HANYA berisi {len(X_test)} hari data uji (testing),")
print(f"      yaitu periode {tgl_test.min().date()} s.d. {tgl_test.max().date()}.")
print(f"      Setiap hari ditampilkan error absolutnya (AbsError),")
print(f"      lalu di baris paling bawah ditambahkan HASIL AKHIR (RMSE & R² agregat).")

# --- Error absolut per data (bermakna dihitung per baris, beda dengan RMSE/R2) ---
abs_error_lr = np.abs(y_pred_lr - y_test.values)
abs_error_rf = np.abs(y_pred_rf - y_test.values)

# Urutkan dulu berdasarkan tanggal (kronologis), baru diberi nomor "Data ke-"
urutan = np.argsort(tgl_test.values)

hasil_harian = pd.DataFrame({
    'Data_ke'         : [f"Data ke-{i+1}" for i in range(len(urutan))],
    'Data_Aktual_mm'  : y_test.values[urutan].round(3),
    'Prediksi_LR_mm'  : y_pred_lr[urutan].round(3),
    'Prediksi_RF_mm'  : y_pred_rf[urutan].round(3),
    'AbsError_LR_mm'  : abs_error_lr[urutan].round(3),
    'AbsError_RF_mm'  : abs_error_rf[urutan].round(3),
})

# --- Baris HASIL AKHIR (RMSE & R2 asli, hanya bermakna dihitung sekali dari seluruh data uji) ---
baris_akhir = pd.DataFrame([{
    'Data_ke'         : 'HASIL AKHIR (RMSE & R2)',
    'Data_Aktual_mm'  : np.nan,
    'Prediksi_LR_mm'  : np.nan,
    'Prediksi_RF_mm'  : np.nan,
    'AbsError_LR_mm'  : f"RMSE={metrics_lr['RMSE']:.4f} | R2={metrics_lr['R²']:.4f}",
    'AbsError_RF_mm'  : f"RMSE={metrics_rf['RMSE']:.4f} | R2={metrics_rf['R²']:.4f}",
}])

hasil_harian_final = pd.concat([hasil_harian, baris_akhir], ignore_index=True)

hasil_harian_final.to_csv('perbandingan_data_uji_vs_aktual.csv', index=False, encoding='utf-8-sig')
print("    ✓ Disimpan → perbandingan_data_uji_vs_aktual.csv")
print(f"    ✓ Total baris  : {len(hasil_harian)} hari data uji + 1 baris HASIL AKHIR")
print(f"    ✓ Kolom        : {list(hasil_harian_final.columns)}")

# Visualisasi
print("\n[8] Membuat Visualisasi ...")

idx_sorted     = np.argsort(tgl_test.values)
y_test_sorted  = y_test.values[idx_sorted]
pred_lr_sorted = y_pred_lr[idx_sorted]
pred_rf_sorted = y_pred_rf[idx_sorted]
x_axis         = range(len(y_test_sorted))

# Linear Regression vs Aktual 
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(x_axis, y_test_sorted, color='black', lw=1.5, label='Aktual', alpha=0.9, zorder=3)
ax.plot(x_axis, pred_lr_sorted, color='steelblue', lw=1.3, linestyle='--',
        label=f'Linear Regression (R²={metrics_lr["R²"]:.4f})', alpha=0.9)
ax.set_xlabel('Index Data Testing (urut waktu)')
ax.set_ylabel('Curah Hujan (mm)')
ax.set_title('Linear Regression vs Aktual\n(BMKG Tanjung Perak Surabaya)', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('lr_vs_aktual.png', bbox_inches='tight')
plt.show()
print("    ✓ Disimpan → lr_vs_aktual.png")

# Random Forest vs Aktual
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(x_axis, y_test_sorted, color='black', lw=1.5, label='Aktual', alpha=0.9, zorder=3)
ax.plot(x_axis, pred_rf_sorted, color='darkorange', lw=1.3, linestyle='-.',
        label=f'Random Forest (R²={metrics_rf["R²"]:.4f})', alpha=0.9)
ax.set_xlabel('Index Data Testing (urut waktu)')
ax.set_ylabel('Curah Hujan (mm)')
ax.set_title('Random Forest vs Aktual\n(BMKG Tanjung Perak Surabaya)', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('rf_vs_aktual.png', bbox_inches='tight')
plt.show()
print("    ✓ Disimpan → rf_vs_aktual.png")

# Gabungan LR vs RF vs Aktual
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(x_axis, y_test_sorted, color='black', lw=1.5, label='Aktual', alpha=0.9, zorder=3)
ax.plot(x_axis, pred_lr_sorted, color='steelblue', lw=1.2, linestyle='--',
        label=f'Linear Regression (R²={metrics_lr["R²"]:.4f})', alpha=0.85)
ax.plot(x_axis, pred_rf_sorted, color='darkorange', lw=1.2, linestyle='-.',
        label=f'Random Forest (R²={metrics_rf["R²"]:.4f})', alpha=0.85)
ax.set_xlabel('Index Data Testing (urut waktu)')
ax.set_ylabel('Curah Hujan (mm)')
ax.set_title('Perbandingan Aktual vs Prediksi: LR vs RF\n'
             '(BMKG Tanjung Perak Surabaya)', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('perbandingan_aktual_vs_prediksi.png', bbox_inches='tight')
plt.show()
print("    ✓ Disimpan → perbandingan_aktual_vs_prediksi.png")

# Scatter Plot Aktual vs Prediksi 
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Prediksi vs Aktual – Curah Hujan (mm)\n(BMKG Tanjung Perak Surabaya)',
             fontsize=13, fontweight='bold')
for ax, y_pred, name, color in zip(
        axes, [y_pred_lr, y_pred_rf],
        ['Linear Regression', 'Random Forest Regression'], ['steelblue', 'darkorange']):
    r2_val = r2_score(y_test, y_pred)
    ax.scatter(y_test, y_pred, alpha=0.55, s=25, color=color,
               edgecolors='k', linewidths=0.2, label=f'R² = {r2_val:.4f}')
    lims = [min(y_test.min(), y_pred.min()) - 1, max(y_test.max(), y_pred.max()) + 1]
    ax.plot(lims, lims, 'r--', lw=1.5, label='Garis ideal (y=x)')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Nilai Aktual (mm)'); ax.set_ylabel('Nilai Prediksi (mm)')
    ax.set_title(name); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_actual_vs_predicted.png', bbox_inches='tight')
plt.show()
print("    ✓ Disimpan → scatter_actual_vs_predicted.png")

# Perbandingan Metrik RMSE & R²
fig, axes = plt.subplots(1, 2, figsize=(11, 6))
fig.suptitle('Perbandingan Metrik Evaluasi\nLinear Regression vs Random Forest',
             fontsize=13, fontweight='bold')
for ax, metric in zip(axes, ['RMSE', 'R²']):
    vals = [metrics_lr[metric], metrics_rf[metric]]
    b = ax.bar(['Linear\nRegression', 'Random\nForest'], vals,
               color=['#2196F3', '#FF9800'], edgecolor='k', lw=0.5, width=0.5)
    for bar, v in zip(b, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(metric, fontweight='bold')
    ax.set_ylabel('Nilai')
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('perbandingan_metrik.png', bbox_inches='tight')
plt.show()
print("    ✓ Disimpan → perbandingan_metrik.png")

# Interpretasi & Kesimpulan
print("\n" + "=" * 65)
print("  INTERPRETASI & KESIMPULAN ")
print("=" * 65)

importances_rf = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS)
top5 = importances_rf.sort_values(ascending=False).head(5)
print("\n📌 5 Fitur Paling Berpengaruh (Random Forest):")
for rank, (feat, imp) in enumerate(top5.items(), 1):
    print(f"   {rank}. {LABEL_MAP.get(feat, feat):30s} → importance = {imp:.4f}")

lr_r2, rf_r2       = metrics_lr['R²'],   metrics_rf['R²']
lr_rmse, rf_rmse   = metrics_lr['RMSE'], metrics_rf['RMSE']

print(f"\n📊 Ringkasan Metrik:")
print(f"   {'Metrik':<8}  {'Linear Regression':>18}  {'Random Forest':>15}")
print(f"   {'-'*46}")
for key in ['RMSE', 'MAE', 'R²']:
    print(f"   {key:<8}  {metrics_lr[key]:>18.4f}  {metrics_rf[key]:>15.4f}")

print("\n📝 Kesimpulan Otomatis:")
print("-" * 65)
all_models = {
    'Linear Regression': (lr_r2, lr_rmse),
    'Random Forest'     : (rf_r2, rf_rmse),
}
best_model_name = max(all_models, key=lambda k: all_models[k][0])
best_r2, best_rmse = all_models[best_model_name]
print(f"  ✅ Model TERBAIK: {best_model_name}")
print(f"     • R²   : {best_r2:.4f}")
print(f"     • RMSE : {best_rmse:.4f}")

if best_r2 >= 0.75:
    kualitas = "BAIK – model mampu menjelaskan variabilitas curah hujan"
elif best_r2 >= 0.50:
    kualitas = "CUKUP – pola umum tertangkap, namun masih ada error signifikan"
else:
    kualitas = "LEMAH – pertimbangkan fitur tambahan atau transformasi data"

print(f"\n  📈 Kualitas model terbaik (R² = {best_r2:.4f}): {kualitas}.")
print(f"\n  🌧  Variabel '{LABEL_MAP.get(top5.index[0], top5.index[0])}' adalah faktor")
print(f"     PALING BERPENGARUH terhadap prediksi curah hujan di Tanjung Perak Surabaya.")
print("\n  ✅ Program selesai. 6 grafik & 2 file CSV telah disimpan:")
print("     1. korelasi_variabel.png")
print("     2. lr_vs_aktual.png")
print("     3. rf_vs_aktual.png")
print("     4. perbandingan_aktual_vs_prediksi.png")
print("     5. scatter_actual_vs_predicted.png")
print("     6. perbandingan_metrik.png")
print("     7. perbandingan_data_uji_vs_aktual.csv")
print("=" * 65)