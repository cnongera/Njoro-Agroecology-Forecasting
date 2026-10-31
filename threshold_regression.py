#!/usr/bin/env python3
"""
Threshold regression: Njoro OND rainfall anomaly vs post-harvest maize price change.

Splits the 11 overlapping years (2015-2026) into WET and DRY/NORMAL regimes
and fits separate slopes. Tests whether the Wainwright threshold mechanism
is visible in the data.
"""

import numpy as np
import pandas as pd
import xarray as xr

# ------------------------------------------------------------------------------
# Configuration — edit if your filenames differ
# ------------------------------------------------------------------------------
WB_CSV    = "WB_KEN_RTFP_mkt_2007_2026-10-05.csv"
CHIRPS_NC = "05a42685-35ad-40e8-ae38-ae606d6cae26.nc"
MARKET    = "Nakuru"
MAIZE_COL = "c_maize_fao"
NJORO_LAT, NJORO_LON = -0.33, 35.94

# ------------------------------------------------------------------------------
# STEP 1 — Load maize prices from the World Bank RTFP file
# ------------------------------------------------------------------------------
print("=" * 70)
print("STEP 1 — Load maize prices")
print("=" * 70)

wb = pd.read_csv(WB_CSV, low_memory=False)
wb = wb[wb["ISO3"] == "KEN"].copy()

mask = pd.Series(False, index=wb.index)
for col in ["adm1_name", "adm2_name", "mkt_name"]:
    if col in wb.columns:
        mask = mask | wb[col].astype(str).str.contains(MARKET, case=False, na=False)
wb = wb[mask]
print(f"Nakuru rows: {len(wb):,}")

wb["Date"] = pd.to_datetime(wb["price_date"], errors="coerce")
wb = wb.dropna(subset=["Date"])
wb["Price_KES"] = pd.to_numeric(wb[MAIZE_COL], errors="coerce") * 90.0
wb = wb.dropna(subset=["Price_KES"])
wb["Year"] = wb["Date"].dt.year
wb["Month"] = wb["Date"].dt.month

# Nov/Jun log price ratio per year
rows = []
for yr, g in wb.groupby("Year"):
    p_jun = g[g["Month"] == 6]["Price_KES"].mean()
    p_nov = g[g["Month"] == 11]["Price_KES"].mean()
    if pd.notna(p_jun) and pd.notna(p_nov) and p_jun > 0 and p_nov > 0:
        rows.append({"Year": int(yr),
                     "log_ratio": float(np.log(p_nov / p_jun))})
price_yearly = pd.DataFrame(rows)
print(f"Years with both Jun and Nov prices: {len(price_yearly)}")

# ------------------------------------------------------------------------------
# STEP 2 — Load CHIRPS rainfall and compute OND anomaly per year
# ------------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 2 — Load CHIRPS rainfall")
print("=" * 70)

ds = xr.open_dataset(CHIRPS_NC)
if "lat" in ds.dims or "lat" in ds.coords:
    ds = ds.rename({"lat": "latitude"})
if "lon" in ds.dims or "lon" in ds.coords:
    ds = ds.rename({"lon": "longitude"})

point = ds.sel(latitude=NJORO_LAT, longitude=NJORO_LON, method="nearest")
print(f"Nearest grid cell: ({float(point['latitude'].values):.3f}, "
      f"{float(point['longitude'].values):.3f})")

var_name = next((v for v in ["precipitation_amount", "precip", "prcp", "rainfall"]
                 if v in ds.data_vars), list(ds.data_vars)[0])
ts = point[var_name].to_pandas()
if isinstance(ts, pd.DataFrame):
    ts = ts.squeeze()

rain = pd.DataFrame({"Date": pd.to_datetime(ts.index), "rain_mm": ts.values})
rain = rain.set_index("Date").resample("MS")["rain_mm"].sum().reset_index()
rain["Year"] = rain["Date"].dt.year
rain["Month"] = rain["Date"].dt.month

ond = rain[rain["Month"].isin([10, 11, 12])]
ond_yearly = ond.groupby("Year")["rain_mm"].sum().reset_index()
ond_yearly.columns = ["Year", "OND_total_mm"]
baseline = ond_yearly["OND_total_mm"].mean()
ond_yearly["OND_anomaly_mm"] = ond_yearly["OND_total_mm"] - baseline
print(f"OND baseline (record-period mean): {baseline:.1f} mm")

# ------------------------------------------------------------------------------
# STEP 3 — Merge on Year
# ------------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 3 — Merge prices and rainfall")
print("=" * 70)

merged = pd.merge(price_yearly, ond_yearly, on="Year").dropna()
merged = merged.sort_values("Year").reset_index(drop=True)
print(f"Overlapping years: {len(merged)}")
print(merged[["Year", "OND_anomaly_mm", "log_ratio"]].to_string(index=False))

# ------------------------------------------------------------------------------
# STEP 4 — Threshold regression
# ------------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 4 — Threshold regression")
print("=" * 70)

def fit_split(df, thr, label_wet, label_dry):
    wet = df[df["OND_anomaly_mm"] >= thr]
    dry = df[df["OND_anomaly_mm"] < thr]

    print(f"\n--- Threshold = {thr:+.0f} mm ---")
    print(f"  {label_wet}: n = {len(wet)}")
    print(f"  {label_dry}: n = {len(dry)}")

    beta_w = beta_d = None

    if len(wet) >= 3:
        beta_w, alpha_w = np.polyfit(wet["OND_anomaly_mm"], wet["log_ratio"], 1)
        resid = wet["log_ratio"] - (alpha_w + beta_w * wet["OND_anomaly_mm"])
        s2 = (resid ** 2).sum() / max(len(wet) - 2, 1)
        se_w = np.sqrt(s2 / ((wet["OND_anomaly_mm"] - wet["OND_anomaly_mm"].mean()) ** 2).sum())
        t_w = beta_w / se_w if se_w > 0 else np.nan
        pct_w = (np.exp(beta_w * 100) - 1) * 100
        print(f"  WET slope:     beta = {beta_w:+.6f} (SE {se_w:.6f}, t = {t_w:+.2f})")
        print(f"                 +100 mm -> {pct_w:+.2f}%")
    else:
        print(f"  WET regime: too few points")

    if len(dry) >= 3:
        beta_d, alpha_d = np.polyfit(dry["OND_anomaly_mm"], dry["log_ratio"], 1)
        resid = dry["log_ratio"] - (alpha_d + beta_d * dry["OND_anomaly_mm"])
        s2 = (resid ** 2).sum() / max(len(dry) - 2, 1)
        se_d = np.sqrt(s2 / ((dry["OND_anomaly_mm"] - dry["OND_anomaly_mm"].mean()) ** 2).sum())
        t_d = beta_d / se_d if se_d > 0 else np.nan
        pct_d = (np.exp(beta_d * 100) - 1) * 100
        print(f"  DRY slope:     beta = {beta_d:+.6f} (SE {se_d:.6f}, t = {t_d:+.2f})")
        print(f"                 +100 mm -> {pct_d:+.2f}%")
    else:
        print(f"  DRY regime: too few points")

    if beta_w is not None and beta_d is not None:
        if beta_w > 0 and beta_d < 0:
            print("  >> SIGN PATTERN SUPPORTS Wainwright threshold mechanism")
        elif beta_w < 0 and beta_d < 0:
            print("  >> Both regimes negative: supply effect dominates")
        elif beta_w > 0 and beta_d > 0:
            print("  >> Both regimes positive: drought-driven price spikes")
        else:
            print("  >> Mixed pattern: inconclusive")

fit_split(merged, 0.0,   "Wet (anomaly >= 0)",      "Dry (< 0)")
fit_split(merged, 50.0,  "Extreme wet (>= +50)",    "Normal/dry (< +50)")
fit_split(merged, 100.0, "Extreme wet (>= +100)",   "Normal/dry (< +100)")

# ------------------------------------------------------------------------------
# STEP 5 — Full-sample regression for reference
# ------------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 5 — Full-sample regression (for comparison)")
print("=" * 70)

x = merged["OND_anomaly_mm"].values
y = merged["log_ratio"].values
beta_full, alpha_full = np.polyfit(x, y, 1)
resid = y - (alpha_full + beta_full * x)
s2 = (resid ** 2).sum() / (len(x) - 2)
se_full = np.sqrt(s2 / ((x - x.mean()) ** 2).sum())
t_full = beta_full / se_full
print(f"Full-sample slope: beta = {beta_full:+.6f} (SE {se_full:.6f}, t = {t_full:+.2f})")

print()
print("=" * 70)
print("DONE")
print("=" * 70)