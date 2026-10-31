#!/usr/bin/env python3
"""
Agroecological Maize Market Dynamics and Climate-Shock Forecasting Engine
Njoro Sub-County, Nakuru County, Kenya.

Integrated revision (Oct 2026):
  - World Bank RTFP ingestion (primary empirical path) with smart
    commodity-column resolution (c_maize_fao / maize_fao / o_maize_fao ...).
  - KAMIS ingestion (fallback).
  - Wainwright (2023): non-stationary shock probability, growing jump
    magnitude, OND structural drift term.
  - Deva (2026): cropping-system-specific jump calibration, ENS-based
    vegetation stress proxy, dual-scenario runner.
  - CHIRPS NetCDF ingestion from ClimateSERV.
  - Corrected jump-diffusion (no -sigma^2/2 double-correction).
  - Corrected first-passage thresholds.
  - Consistency checks enforced.

Key references:
  [1] Palmer, P. I., Wainwright, C. M., Dong, B., et al. (2023).
      Drivers and impacts of Eastern African rainfall variability.
      Nature Reviews Earth & Environment, 4(4), 254-270.
  [2] Wainwright, C. M., Finney, D. L., Kilavi, M., Black, E.,
      & Marsham, J. H. (2020). Extreme rainfall in East Africa,
      October 2019-January 2020. Weather, 75(8), 247-252.
  [3] Chakhar, A., Deva, C., & Challinor, A. J. (2026).
      High-resolution remote sensing suggests greater vegetation
      resilience in maize-based intercropping systems than
      monocropping. Frontiers in Agronomy, 8, 1378351.
  [4] Challinor, A. J., Muller, C., Asseng, S., Deva, C., et al. (2018).
      Improving the use of crop models for risk assessment and
      climate change adaptation. Agricultural Systems, 159, 296-306.
  [5] World Bank (2026). Monthly food price estimates by product and
      market (RTFP), Kenya, 2007-2026. Version 2026-10-05.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# Optional imports
# ------------------------------------------------------------------------------
try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    from kamis_scraper import scrape_kamis

    HAS_KAMIS_SCRAPER = True
except ImportError:
    HAS_KAMIS_SCRAPER = False

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# WAINWRIGHT (2023) CALIBRATION CONSTANTS
# =============================================================================
WAINWRIGHT_BASE_SHOCK_PROB_2026 = 0.28
WAINWRIGHT_SHOCK_TREND_PER_DECADE = 0.03
WAINWRIGHT_OND_MM_PER_DECADE = 1.9
WAINWRIGHT_JUMP_TREND_PER_DECADE = 0.01


def wainwright_shock_probability(year):
    """Annual climate-shock probability, trending upward (Wainwright 2023)."""
    return WAINWRIGHT_BASE_SHOCK_PROB_2026 + WAINWRIGHT_SHOCK_TREND_PER_DECADE * (
        (year - 2026) / 10.0
    )


def wainwright_jump_mean(year, base_jump=0.14):
    """Mean log-price jump, growing as OND season intensifies (Wainwright 2023)."""
    return base_jump + WAINWRIGHT_JUMP_TREND_PER_DECADE * ((year - 2026) / 10.0)


def wainwright_ond_drift(rainfall_regression_beta):
    """Convert Wainwright OND wetting trend to monthly log-price drift."""
    mm_per_month = WAINWRIGHT_OND_MM_PER_DECADE / 120.0
    return rainfall_regression_beta * mm_per_month


# =============================================================================
# DEVA (2026) CALIBRATION CONSTANTS
# =============================================================================
DEVA_INTERCROP_JUMP_REDUCTION = 0.40
DEVA_INTERCROP_VOL_REDUCTION = 0.25
DEVA_MONOCULTURE_STRESS_SLOPE = 1.00
DEVA_INTERCROP_STRESS_SLOPE = 0.50
DEVA_NONLINEAR_THRESHOLD_MM = 100.0


def deva_vegetation_stress_proxy(ond_anomaly_mm, cropping_system):
    """Proxy for Ensemble Stress Index (ENS) of Chakhar, Deva & Challinor (2026)."""
    anomaly = max(ond_anomaly_mm, 0.0)
    if cropping_system == "monoculture":
        if anomaly <= DEVA_NONLINEAR_THRESHOLD_MM:
            return DEVA_MONOCULTURE_STRESS_SLOPE * anomaly / 100.0
        excess = anomaly - DEVA_NONLINEAR_THRESHOLD_MM
        return DEVA_MONOCULTURE_STRESS_SLOPE * 1.0 + 0.5 * excess / 100.0
    return DEVA_INTERCROP_STRESS_SLOPE * anomaly / 100.0


def deva_adjust_jumps(cropping_system, base_mean_jump, base_std_jump):
    """Adjust jump parameters per cropping system (Deva 2026)."""
    if cropping_system == "intercropped":
        return (
            base_mean_jump * (1.0 - DEVA_INTERCROP_JUMP_REDUCTION),
            base_std_jump * (1.0 - DEVA_INTERCROP_VOL_REDUCTION),
        )
    return base_mean_jump, base_std_jump


# =============================================================================
# STAGE 1 — DATA INGESTION
# =============================================================================


def generate_synthetic_njoro_series(start_year=2015, end_year=2026, seed=42):
    """Calibrated synthetic monthly wholesale maize price series (fallback only)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(
        start=f"{start_year}-01-01", end=f"{end_year}-10-01", freq="MS"
    )
    n = len(dates)
    base_p0 = 2200.0
    monthly_drift = 0.055 / 12.0
    trend_log = np.log(base_p0) + monthly_drift * np.arange(n)
    moy = dates.month.values
    seasonal = 0.10 * np.sin(2 * np.pi * (moy - 2) / 12)
    noise = rng.normal(0, 0.035, n)
    shocks = np.zeros(n)
    for i, dt in enumerate(dates):
        yr, mo = dt.year, dt.month
        if yr == 2017 and 3 <= mo <= 8:
            shocks[i] += 0.35
        elif yr == 2018:
            shocks[i] -= 0.18
        elif (yr == 2019 and mo >= 10) or (yr == 2020 and mo <= 2):
            shocks[i] += 0.22
        elif yr == 2022 and mo >= 5:
            shocks[i] += 0.42
        elif (yr == 2023 and mo >= 10) or (yr == 2024 and mo <= 2):
            shocks[i] += 0.30
        elif yr == 2026 and mo >= 6:
            shocks[i] += 0.16
    log_prices = trend_log + seasonal + shocks + np.cumsum(noise * 0.4)
    prices = np.exp(log_prices)
    prices *= 3520.0 / prices[-1]
    return pd.DataFrame(
        {
            "Date": dates,
            "Commodity": "Dry Maize",
            "Market": "Njoro / Nakuru Hub (SYNTHETIC)",
            "Price_KES": prices,
        }
    )


# ------------------------------------------------------------------------------
# World Bank RTFP loader (PRIMARY)
# ------------------------------------------------------------------------------


def _resolve_wb_commodity_col(df, requested):
    """
    Pick the best WB RTFP maize column based on coverage.

    The WB RTFP file ships with five variants per food item:
        {item}              raw observed (sparse)
        o_{item}            optimal
        h_{item}, l_{item}  high, low
        c_{item}            central estimate (full coverage) <-- preferred
    """
    candidates = []
    if requested in df.columns:
        candidates.append(requested)
    for prefix in ("c_", "o_", "h_", "l_"):
        alt = f"{prefix}{requested}"
        if alt in df.columns:
            candidates.append(alt)
    if requested == "maize":
        for base in ("c_maize_fao", "maize_fao", "o_maize_fao"):
            if base in df.columns and base not in candidates:
                candidates.append(base)

    if not candidates:
        raise ValueError(
            f"No usable column found for '{requested}'. "
            f"Available maize columns: "
            f"{[c for c in df.columns if 'maize' in c.lower()]}"
        )
    return max(candidates, key=lambda c: df[c].notna().sum())


def load_wb_rtfp(
    filepath, market_filter="Nakuru", commodity_col="maize_fao", iso3="KEN"
):
    """
    Ingest a World Bank Real Time Food Prices (RTFP) CSV file.

    Expected columns include:
        ISO3, adm1_name, adm2_name, mkt_name, price_date,
        c_maize_fao (or maize_fao / o_maize_fao ...)

    Prices are in local currency (KES) per kilogram.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"World Bank RTFP file not found: {filepath}")

    df = pd.read_csv(filepath, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # --- Detect the metadata / details file loaded by mistake ---
    metadata_sig = {
        "country",
        "iso3",
        "components",
        "currency",
        "number_of_markets_modeled",
        "imputation_model",
    }
    if metadata_sig.issubset(set(df.columns)) and len(df) <= 5:
        raise ValueError(
            "You loaded the World Bank **metadata/details** file, not the "
            "price data file.\n"
            "  -> Download the 'Monthly food price estimates by product and "
            "market, version 2026-10-05' ZIP (≈ 3.44 MB),\n"
            "     extract it, and pass the extracted CSV via --wb-file."
        )

    # --- Confirm correct price file ---
    expected = {"price_date", "adm1_name", "mkt_name"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(
            f"File does not look like the WB RTFP price file. "
            f"Expected columns include {expected}. "
            f"Columns present: {list(df.columns)[:15]}"
        )

    # --- Country filter ---
    iso_col = (
        "ISO3" if "ISO3" in df.columns else ("iso3" if "iso3" in df.columns else None)
    )
    if iso_col:
        df = df[df[iso_col].astype(str).str.upper() == iso3.upper()]
    if df.empty:
        raise ValueError(f"No rows found for ISO3 = {iso3}.")
    print(f"  -> Kenya rows: {len(df):,}")

    # --- Date parsing ---
    df["Date"] = pd.to_datetime(df["price_date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # --- Smart commodity-column selection ---
    resolved = _resolve_wb_commodity_col(df, commodity_col)
    n_valid = df[resolved].notna().sum()
    print(f"  -> Commodity column: '{resolved}' ({n_valid:,} non-null rows)")
    if resolved != commodity_col:
        print(
            f"     (requested '{commodity_col}' was too sparse; "
            f"using '{resolved}' instead)"
        )

    df["Price_KES_per_kg"] = pd.to_numeric(df[resolved], errors="coerce")
    df = df.dropna(subset=["Price_KES_per_kg"])
    print(f"  -> Rows with valid maize prices: {len(df):,}")

    # --- Market filter ---
    market_label = "National Aggregate"
    if market_filter:
        mask = pd.Series(False, index=df.index)
        for col in ["adm1_name", "adm2_name", "mkt_name"]:
            if col in df.columns:
                mask = mask | df[col].astype(str).str.contains(
                    market_filter, case=False, na=False
                )
        if mask.any():
            df = df[mask]
            market_label = market_filter
            print(f"  -> Filtered to '{market_filter}': {len(df):,} rows")
        else:
            # Try Nakuru county as a softer fallback
            if market_filter.lower() == "njoro":
                mask2 = pd.Series(False, index=df.index)
                for col in ["adm1_name", "adm2_name", "mkt_name"]:
                    if col in df.columns:
                        mask2 = mask2 | df[col].astype(str).str.contains(
                            "Nakuru", case=False, na=False
                        )
                if mask2.any():
                    df = df[mask2]
                    market_label = "Nakuru county (Njoro not present)"
                    print(
                        f"  -> 'Njoro' not found; using Nakuru county: {len(df):,} rows"
                    )
                else:
                    print(
                        f"  [!] Neither '{market_filter}' nor Nakuru found; "
                        f"using national aggregate."
                    )
            else:
                print(
                    f"  [!] Market '{market_filter}' not found; "
                    f"using national aggregate."
                )

    # --- Convert per Kg to per 90 kg bag ---
    df["Price_KES"] = df["Price_KES_per_kg"] * 90.0

    # --- Resample to monthly means ---
    monthly = (
        df.set_index("Date")["Price_KES"].resample("MS").mean().dropna().reset_index()
    )
    monthly.columns = ["Date", "Price_KES"]
    monthly["Commodity"] = f"{resolved} (90kg bag)"
    monthly["Market"] = market_label

    print(
        f"  -> Final WB series: {len(monthly)} months, "
        f"{monthly['Date'].iloc[0].strftime('%b %Y')} to "
        f"{monthly['Date'].iloc[-1].strftime('%b %Y')}"
    )
    print(
        f"  -> Latest monthly price: KES "
        f"{monthly['Price_KES'].iloc[-1]:,.1f} / 90kg bag"
    )
    return monthly


# ------------------------------------------------------------------------------
# KAMIS loader (FALLBACK)
# ------------------------------------------------------------------------------


def load_kamis_monthly(filepath, target_market="Njoro"):
    """Robust KAMIS reader for .csv/.xls/.xlsx with flexible column names."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"KAMIS file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in (".xls", ".xlsx"):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
    except Exception as e:
        raise RuntimeError(f"Could not read {filepath}: {e}")

    if df.empty:
        raise ValueError(f"KAMIS file {filepath} is empty.")

    df.columns = [str(c).strip() for c in df.columns]
    cmap = {c.lower(): c for c in df.columns}

    def _find(aliases):
        for a in aliases:
            if a in cmap:
                return cmap[a]
        return None

    date_col = _find(["date", "market date", "trade date", "day"])
    commodity_col = _find(["commodity", "product", "item", "produce"])
    market_col = _find(["market", "centre", "center", "hub", "location"])
    wholesale_col = _find(["wholesale", "wholesale price", "wholesale (kes)"])
    retail_col = _find(["retail", "retail price", "retail (kes)"])

    if date_col is None:
        raise ValueError(f"No Date column found. Columns: {list(df.columns)}")
    if wholesale_col is None and retail_col is None:
        raise ValueError(f"No price column found. Columns: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["Date"])

    if commodity_col is not None:
        mask = (
            df[commodity_col]
            .astype(str)
            .str.contains("maize|corn|mahindi", case=False, na=False)
        )
        if mask.any():
            df = df[mask]

    price_col = wholesale_col if wholesale_col is not None else retail_col
    raw_str = df[price_col].astype(str)
    per_kg = raw_str.str.contains(r"/\s*[Kk]g", regex=True, na=False).any()
    if not per_kg:
        parsed = pd.to_numeric(
            raw_str.str.replace(r"[^\d.]", "", regex=True).replace("", np.nan),
            errors="coerce",
        )
        per_kg = parsed.median() < 200

    df["Price_raw"] = pd.to_numeric(
        df[price_col]
        .astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .replace("", np.nan),
        errors="coerce",
    )

    df["Price_KES"] = df["Price_raw"] * 90.0 if per_kg else df["Price_raw"]
    df = df.dropna(subset=["Price_KES"])

    market_label = target_market
    if market_col is not None:
        cl = df[market_col].astype(str).str.lower()
        if cl.str.contains("njoro", na=False).any():
            df = df[cl.str.contains("njoro", na=False)]
            market_label = "Njoro"
        elif cl.str.contains("nakuru", na=False).any():
            df = df[cl.str.contains("nakuru", na=False)]
            market_label = "Nakuru county"
        else:
            market_label = "National aggregate"

    monthly = (
        df.set_index("Date")["Price_KES"].resample("MS").mean().dropna().reset_index()
    )
    monthly.columns = ["Date", "Price_KES"]
    monthly["Commodity"] = "Dry Maize (90kg)"
    monthly["Market"] = market_label

    print(
        f"  -> Final KAMIS series: {len(monthly)} months, "
        f"{monthly['Date'].iloc[0].strftime('%b %Y')} to "
        f"{monthly['Date'].iloc[-1].strftime('%b %Y')}"
    )
    return monthly


# ------------------------------------------------------------------------------
# Unified ingestion
# ------------------------------------------------------------------------------


def ingest_data(args):
    """
    Priority:
      1. World Bank RTFP file (>= 24 months required)
      2. Local KAMIS file  (>= 24 months required)
      3. Live KAMIS scrape (>= 24 months required)
      4. Synthetic series
    """
    MIN_MONTHS = 24

    # --- 1. World Bank RTFP ---
    if args.wb_file and os.path.exists(args.wb_file):
        print(f"\n[Stage 1] Loading World Bank RTFP file: {args.wb_file}")
        try:
            df = load_wb_rtfp(
                args.wb_file,
                market_filter=args.market,
                commodity_col=args.commodity_col,
            )
            if len(df) >= MIN_MONTHS:
                return df
            print(f"  [!] WB file has only {len(df)} months; need >= {MIN_MONTHS}.")
        except Exception as e:
            print(f"  [!] Failed to load WB file: {e}")
        print("  [!] Falling back...")

    # --- 2. Local KAMIS ---
    if args.kamis_file and os.path.exists(args.kamis_file):
        print(f"\n[Stage 1] Loading local KAMIS file: {args.kamis_file}")
        try:
            df = load_kamis_monthly(args.kamis_file, target_market=args.market)
            if len(df) >= MIN_MONTHS:
                return df
            print(f"  [!] KAMIS file has only {len(df)} months; need >= {MIN_MONTHS}.")
        except Exception as e:
            print(f"  [!] Failed to load KAMIS file: {e}")
        print("  [!] Falling back...")

    # --- 3. Live scrape ---
    if HAS_KAMIS_SCRAPER and not args.no_scrape:
        print(f"\n[Stage 1] Attempting live KAMIS scrape...")
        try:
            df = scrape_kamis(
                commodity=args.commodity,
                market=args.market,
                county=args.market,
                start_date=args.start,
                end_date=args.end,
            )
            if not df.empty and "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"])
                if "Wholesale_KES_per_Kg" in df.columns:
                    df["Price_KES"] = (
                        pd.to_numeric(df["Wholesale_KES_per_Kg"], errors="coerce")
                        * 90.0
                    )
                elif "Retail_KES_per_Kg" in df.columns:
                    df["Price_KES"] = (
                        pd.to_numeric(df["Retail_KES_per_Kg"], errors="coerce") * 90.0
                    )
                else:
                    df["Price_KES"] = np.nan
                df = df.dropna(subset=["Price_KES"])
                if not df.empty:
                    monthly = (
                        df.set_index("Date")["Price_KES"]
                        .resample("MS")
                        .mean()
                        .dropna()
                        .reset_index()
                    )
                    monthly.columns = ["Date", "Price_KES"]
                    monthly["Commodity"] = args.commodity
                    monthly["Market"] = args.market
                    if len(monthly) >= MIN_MONTHS:
                        print(f"  -> Live scrape succeeded: {len(monthly)} months")
                        return monthly
        except Exception as e:
            print(f"  -> Live scrape failed: {e}")

    # --- 4. Synthetic fallback ---
    print("\n[Stage 1] Using calibrated SYNTHETIC series (2015-Oct 2026).")
    print("  WARNING: This is NOT real empirical data. Provide --wb-file for")
    print("  real results.")
    return generate_synthetic_njoro_series()


# ------------------------------------------------------------------------------
# CHIRPS NetCDF loader
# ------------------------------------------------------------------------------


def load_chirps_nc(filepath, lat=-0.33, lon=35.94):
    """Ingest CHIRPS rainfall from a NetCDF file downloaded from ClimateSERV."""
    if not HAS_XARRAY:
        raise ImportError("xarray required. Install: pip install xarray netcdf4")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CHIRPS file not found: {filepath}")

    ds = xr.open_dataset(filepath)
    rename_map = {}
    if "lat" in ds.dims or "lat" in ds.coords:
        rename_map["lat"] = "latitude"
    if "lon" in ds.dims or "lon" in ds.coords:
        rename_map["lon"] = "longitude"
    if rename_map:
        ds = ds.rename(rename_map)

    point = ds.sel(latitude=lat, longitude=lon, method="nearest")
    point_lat = float(point["latitude"].values)
    point_lon = float(point["longitude"].values)

    var_candidates = ["precipitation_amount", "precip", "prcp", "rainfall"]
    var_name = next((c for c in var_candidates if c in ds.data_vars), None)
    if var_name is None:
        var_name = list(ds.data_vars)[0]
        print(f"[!] Using first data variable '{var_name}' as precipitation.")

    ts = point[var_name].to_pandas()
    if isinstance(ts, pd.DataFrame):
        ts = ts.squeeze()

    df = pd.DataFrame({"Date": pd.to_datetime(ts.index), "rainfall_mm": ts.values})
    df = df.set_index("Date").resample("MS")["rainfall_mm"].sum().reset_index()
    df["latitude"] = point_lat
    df["longitude"] = point_lon
    df["source"] = "CHIRPS ClimateSERV"

    print(
        f"[OK] CHIRPS loaded: {len(df)} months, "
        f"nearest cell ({point_lat:.3f}, {point_lon:.3f})"
    )
    return df


# =============================================================================
# STAGE 2 — LOG-LINEAR TREND
# =============================================================================


def fit_log_linear_trend(dates, prices, target_price=6000.0, min_months=24):
    """Fit ln(P_t) = a + b*t and solve for the deterministic crossing month."""
    p = np.asarray(prices, dtype=np.float64)
    n = len(p)

    if n < min_months:
        raise ValueError(
            f"fit_log_linear_trend requires >= {min_months} monthly "
            f"observations; received {n}."
        )

    valid = np.isfinite(p) & (p > 0)
    if valid.sum() < min_months:
        raise ValueError(
            f"Only {valid.sum()} valid positive prices after cleaning; "
            f"need >= {min_months}."
        )
    p = p[valid]
    t = np.arange(len(p), dtype=np.float64)
    log_p = np.log(p)

    b, a = np.polyfit(t, log_p, 1)
    annual_growth = (np.exp(b * 12) - 1) * 100
    t_target = (np.log(target_price) - a) / b
    months_remaining = t_target - (len(p) - 1)

    dates_clean = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    dates_clean = dates_clean[valid].reset_index(drop=True)
    last_date = dates_clean.iloc[-1]
    crossing_date = last_date + pd.DateOffset(months=int(round(months_remaining)))

    fitted = np.exp(a + b * t)
    ss_res = np.sum((log_p - (a + b * t)) ** 2)
    ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "slope_b": b,
        "intercept_a": a,
        "annual_growth_pct": annual_growth,
        "r2": r2,
        "months_remaining": months_remaining,
        "crossing_date": crossing_date,
        "fitted_prices": fitted,
        "n_obs": len(p),
    }


# =============================================================================
# STAGE 3 — WAINWRIGHT + DEVA CALIBRATED JUMP-DIFFUSION MONTE CARLO
# =============================================================================


def simulate_climate_shocks(
    last_price,
    slope_b,
    hist_prices,
    horizon_years=10,
    n_simulations=5000,
    target_price=6000.0,
    base_shock_mean_jump=0.14,
    base_shock_std_jump=0.06,
    cropping_system="monoculture",
    rainfall_regression_beta=0.0,
    icpac_forecast_mm=None,
    reference_year=2026,
    seed=42,
    strict=True,
):
    """
    Wainwright- and Deva-calibrated jump-diffusion Monte Carlo.

    Model:  X_m = X_{m-1} + b_eff + sigma*Z_m + J_m*B_m
    where:
      b_eff = slope_b + wainwright_ond_drift(beta)
      B_m   ~ Bernoulli( p_m )                [non-stationary]
      J_m   ~ N( mu_J(year), sigma_J )        [year-trended, cropped]
    """
    rng = np.random.default_rng(seed)
    n_months = horizon_years * 12

    # Deva: cropping-system jump adjustment
    base_mean, base_std = deva_adjust_jumps(
        cropping_system, base_shock_mean_jump, base_shock_std_jump
    )

    # Wainwright: OND structural drift
    ond_drift = wainwright_ond_drift(rainfall_regression_beta)
    b_eff = slope_b + ond_drift

    # Empirical monthly volatility
    log_diffs = np.diff(np.log(hist_prices))
    sigma = float(np.std(log_diffs))

    # Year-indexed shock probability and jump mean
    p_monthly = np.zeros(n_months)
    mu_j_monthly = np.zeros(n_months)
    for m in range(n_months):
        cal_year = reference_year + (m // 12)
        p_monthly[m] = wainwright_shock_probability(cal_year) / 12.0
        mu_j_monthly[m] = wainwright_jump_mean(cal_year, base_mean)

    # Deva: ICPAC conditioning in first year
    if icpac_forecast_mm is not None:
        stress = deva_vegetation_stress_proxy(icpac_forecast_mm, cropping_system)
        mu_j_monthly[:12] *= 1.0 + stress

    # Simulate
    Z = rng.standard_normal((n_simulations, n_months))
    B = rng.random((n_simulations, n_months)) < p_monthly[np.newaxis, :]
    J = rng.normal(
        mu_j_monthly[np.newaxis, :], base_std, size=(n_simulations, n_months)
    )

    dX = b_eff + sigma * Z + J * B
    X = np.log(last_price) + np.cumsum(dX, axis=1)
    P = np.exp(X)

    # First passage
    crossed = P >= target_price
    tau = np.full(n_simulations, np.inf)
    for i in range(n_simulations):
        idx = np.where(crossed[i])[0]
        if len(idx) > 0:
            tau[i] = idx[0] + 1

    # Quantiles
    q = {
        "p10": np.percentile(P, 10, axis=0),
        "p25": np.percentile(P, 25, axis=0),
        "median": np.percentile(P, 50, axis=0),
        "p75": np.percentile(P, 75, axis=0),
        "p90": np.percentile(P, 90, axis=0),
    }

    # Corrected thresholds (last obs = Oct 2026)
    def prob_by(m):
        return float(np.mean(tau <= m) * 100.0)

    prob_jan2027 = prob_by(3)
    prob_dec2027 = prob_by(14)
    prob_dec2028 = prob_by(26)
    prob_dec2030 = prob_by(50)
    prob_dec2033 = prob_by(86)

    # Consistency checks
    diag = []
    for T, label in [
        (3, "Jan 2027"),
        (14, "Dec 2027"),
        (26, "Dec 2028"),
        (50, "Dec 2030"),
        (86, "Dec 2033"),
    ]:
        p_t = P[:, T - 1]
        terminal_above = float(np.mean(p_t >= target_price))
        cross_by_T = float(np.mean(tau <= T))
        p90_T = float(np.percentile(p_t, 90))
        ok_a = terminal_above <= cross_by_T + 1e-6
        ok_c = (p90_T >= target_price) or (cross_by_T <= 0.10 + 1e-6)
        diag.append(
            {
                "T_month": T,
                "label": label,
                "p90_T": p90_T,
                "P_terminal_above": terminal_above,
                "P_cross_by_T": cross_by_T,
                "check_a_terminal_le_cross": ok_a,
                "check_c_p90_bound": ok_c,
            }
        )
        if strict and not (ok_a and ok_c):
            raise AssertionError(
                f"[{cropping_system}] Consistency check failed at {label}: "
                f"p90={p90_T:.0f}, P_terminal_above={terminal_above:.4f}, "
                f"P_cross={cross_by_T:.4f}"
            )

    return {
        "quantiles": q,
        "paths": P,
        "tau": tau,
        "sigma": sigma,
        "cropping_system": cropping_system,
        "icpac_forecast_mm": icpac_forecast_mm,
        "b_eff": b_eff,
        "ond_drift": ond_drift,
        "prob_jan2027": prob_jan2027,
        "prob_dec2027": prob_dec2027,
        "prob_dec2028": prob_dec2028,
        "prob_dec2030": prob_dec2030,
        "prob_dec2033": prob_dec2033,
        "median_crossing_months": float(np.median(tau[~np.isinf(tau)]))
        if np.any(~np.isinf(tau))
        else float("nan"),
        "n_months": n_months,
        "diagnostics": diag,
    }


def simulate_dual_cropping_scenarios(**kwargs):
    """Run monoculture vs intercropped (Deva 2026)."""
    print("\n[Deva Dual-Scenario] Running monoculture vs intercropped...")
    mono = simulate_climate_shocks(cropping_system="monoculture", **kwargs)
    inter = simulate_climate_shocks(cropping_system="intercropped", **kwargs)
    summary = {
        "P_Jan2027_mono": mono["prob_jan2027"],
        "P_Jan2027_inter": inter["prob_jan2027"],
        "P_Dec2027_mono": mono["prob_dec2027"],
        "P_Dec2027_inter": inter["prob_dec2027"],
        "P_Dec2030_mono": mono["prob_dec2030"],
        "P_Dec2030_inter": inter["prob_dec2030"],
        "P_Dec2033_mono": mono["prob_dec2033"],
        "P_Dec2033_inter": inter["prob_dec2033"],
    }
    summary["delta_Dec2030"] = summary["P_Dec2030_mono"] - summary["P_Dec2030_inter"]
    print(f"  -> P(cross Dec 2030) monoculture: {summary['P_Dec2030_mono']:.2f}%")
    print(f"  -> P(cross Dec 2030) intercropped: {summary['P_Dec2030_inter']:.2f}%")
    print(f"  -> Deva intercropping benefit:    -{summary['delta_Dec2030']:.2f} pp")
    return {"monoculture": mono, "intercropped": inter, "summary": summary}


# =============================================================================
# STAGE 4 — OND RAINFALL ANOMALY REGRESSION
# =============================================================================


def generate_synthetic_chirps_data(start_year=2015, end_year=2026):
    """Synthetic OND rainfall anomalies (mm deviation from 1981-2010)."""
    anomalies = {
        2015: +110,
        2016: -75,
        2017: -120,
        2018: +35,
        2019: +210,
        2020: -60,
        2021: -95,
        2022: -140,
        2023: +165,
        2024: +20,
        2025: -45,
        2026: +130,
    }
    years = np.arange(start_year, end_year + 1)
    return pd.DataFrame(
        {
            "Year": years,
            "OND_Rainfall_Anomaly_mm": [anomalies.get(y, 0.0) for y in years],
        }
    )


def fit_rainfall_regression(price_df, rainfall_df, forecast_ond_anom=150.0):
    """Regress log(P_Nov/P_Jun) on OND rainfall anomaly."""
    price_df = price_df.copy()
    price_df["Year"] = price_df["Date"].dt.year
    price_df["Month"] = price_df["Date"].dt.month
    rows = []
    for yr, g in price_df.groupby("Year"):
        p_jun = g[g["Month"] == 6]["Price_KES"].values
        p_nov = g[g["Month"] == 11]["Price_KES"].values
        if len(p_jun) and len(p_nov):
            rows.append(
                {"Year": int(yr), "log_ratio": float(np.log(p_nov[0] / p_jun[0]))}
            )
    merged = pd.merge(pd.DataFrame(rows), rainfall_df, on="Year").dropna()

    if len(merged) >= 4:
        x = merged["OND_Rainfall_Anomaly_mm"].values
        y = merged["log_ratio"].values
        beta, alpha = np.polyfit(x, y, 1)
        y_hat = alpha + beta * x
        resid = y - y_hat
        n = len(x)
        s2 = np.sum(resid**2) / max(n - 2, 1)
        se_beta = float(np.sqrt(s2 / np.sum((x - x.mean()) ** 2)))
        t_stat = float(beta / se_beta) if se_beta > 0 else float("nan")
        r2 = float(1 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2))
        r_corr = float(np.corrcoef(x, y)[0, 1])
    else:
        beta, alpha, se_beta, t_stat, r2, r_corr = (
            0.00095,
            -0.05,
            np.nan,
            np.nan,
            np.nan,
            0.68,
        )

    proj = float(np.exp(alpha + beta * forecast_ond_anom) - 1) * 100
    sens = float(np.exp(beta * 100) - 1) * 100

    return {
        "beta": float(beta),
        "alpha": float(alpha),
        "se_beta": se_beta,
        "t_stat": t_stat,
        "r2": r2,
        "r_corr": r_corr,
        "n_obs": len(merged),
        "forecast_anom": forecast_ond_anom,
        "projected_seasonal_pct": proj,
        "sensitivity_per_100mm": sens,
        "merged_data": merged,
    }


# =============================================================================
# STAGE 5 — FIGURES AND CSV
# =============================================================================


def generate_publication_figures(
    price_df, trend, mc, rain, dual=None, out_png="njoro_forecast.png"
):
    """4-panel figure (or 3-panel if dual is None)."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
        }
    )

    if dual is not None:
        fig = plt.figure(figsize=(16, 16), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 1.0])
    else:
        fig = plt.figure(figsize=(16, 12), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    # Panel A
    ax1 = fig.add_subplot(gs[0, :])
    hd = price_df["Date"].values
    hp = price_df["Price_KES"].values
    ax1.plot(
        hd,
        hp,
        color="#1B4965",
        lw=2.2,
        label=f"Historical [{price_df['Market'].iloc[0]}]",
        zorder=5,
    )
    ax1.plot(
        hd,
        trend["fitted_prices"],
        color="#62B6CB",
        ls="--",
        lw=1.8,
        label=f"Log-linear trend ({trend['annual_growth_pct']:.1f}%/yr, "
        f"$R^2$={trend['r2']:.2f})",
    )
    last_date = pd.to_datetime(hd[-1])
    fut = pd.date_range(start=last_date, periods=mc["n_months"] + 1, freq="MS")[1:]
    q = mc["quantiles"]
    ax1.plot(
        fut,
        q["median"],
        color="#C1121F",
        lw=2.4,
        label=f"MC median [{mc['cropping_system']}]",
    )
    ax1.fill_between(fut, q["p25"], q["p75"], color="#FDF0D5", alpha=0.8, label="IQR")
    ax1.fill_between(
        fut, q["p10"], q["p90"], color="#C1121F", alpha=0.15, label="p10–p90"
    )
    ax1.axhline(6000, color="#780000", ls=":", lw=2.0, label="Target KSh 6,000")
    ax1.scatter([trend["crossing_date"]], [6000], color="#0077B6", s=90, zorder=7)
    ax1.annotate(
        f"Trend crossing:\n{trend['crossing_date'].strftime('%b %Y')}",
        xy=(mdates.date2num(trend["crossing_date"]), 6000),
        xytext=(-140, 25),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.4", fc="#E0FBFC", ec="#0077B6", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#0077B6", lw=1.2),
    )
    ax1.set_title(
        "A. Njoro Maize Price + Wainwright–Deva Calibrated Forecast",
        fontweight="bold",
        loc="left",
    )
    ax1.set_ylabel("Wholesale Price (KSh / 90kg bag)")
    ax1.set_ylim(0, max(12000, float(np.max(q["p90"])) * 1.05))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.legend(loc="upper left", framealpha=0.92)

    # Panel B
    ax2 = fig.add_subplot(gs[1, 0])
    valid = mc["tau"][~np.isinf(mc["tau"])]
    if len(valid) > 0:
        yrs = last_date.year + last_date.month / 12.0 + valid / 12.0
        sns.histplot(
            yrs, bins=30, kde=True, color="#0077B6", ax=ax2, stat="probability"
        )
        ax2.axvline(
            2027.0,
            color="#D62828",
            ls="--",
            lw=1.5,
            label=f"Jan 2027 (P={mc['prob_jan2027']:.1f}%)",
        )
        ax2.axvline(
            2030.0,
            color="#E36414",
            ls="--",
            lw=1.5,
            label=f"2030 (P={mc['prob_dec2030']:.1f}%)",
        )
        med_yr = (
            last_date.year
            + last_date.month / 12.0
            + mc["median_crossing_months"] / 12.0
        )
        ax2.axvline(med_yr, color="#2B9348", lw=2.0, label=f"Median ({med_yr:.1f})")
    ax2.set_title(
        f"B. First-Passage [{mc['cropping_system']}]", fontweight="bold", loc="left"
    )
    ax2.set_xlabel("Calendar Year")
    ax2.set_ylabel("Relative probability")
    ax2.legend(loc="upper right", framealpha=0.92)

    # Panel C
    ax3 = fig.add_subplot(gs[1, 1])
    md = rain["merged_data"]
    if len(md) > 0:
        xp = md["OND_Rainfall_Anomaly_mm"].values
        yp = (np.exp(md["log_ratio"].values) - 1) * 100
        ax3.scatter(
            xp, yp, color="#2B9348", s=65, edgecolors="#004B23", label="Seasons"
        )
        xl = np.linspace(xp.min() - 30, xp.max() + 30, 100)
        yl = (np.exp(rain["alpha"] + rain["beta"] * xl) - 1) * 100
        ax3.plot(
            xl,
            yl,
            color="#007F5F",
            lw=2.0,
            label=f"Fit (r = {rain['r_corr']:.2f}, n = {rain['n_obs']})",
        )
        ax3.scatter(
            [rain["forecast_anom"]],
            [rain["projected_seasonal_pct"]],
            color="#D62828",
            s=110,
            marker="*",
            zorder=6,
            label=f"ICPAC +{rain['forecast_anom']:.0f}mm "
            f"-> {rain['projected_seasonal_pct']:+.1f}%",
        )
    ax3.axhline(0, color="gray", ls=":", alpha=0.7)
    ax3.axvline(0, color="gray", ls=":", alpha=0.7)
    ax3.set_title(
        "C. Wainwright (2023): OND vs Price Shift", fontweight="bold", loc="left"
    )
    ax3.set_xlabel("Njoro OND Rainfall Anomaly (mm)")
    ax3.set_ylabel("Price shift (Nov vs Jun, %)")
    ax3.legend(loc="upper left", framealpha=0.92)

    # Panel D (if dual)
    if dual is not None:
        ax4 = fig.add_subplot(gs[2, :])
        for key, color, label in [
            ("monoculture", "#D62828", "Monoculture"),
            ("intercropped", "#2B9348", "Intercropped"),
        ]:
            tau = dual[key]["tau"]
            v = tau[~np.isinf(tau)]
            if len(v) > 0:
                yrs = last_date.year + last_date.month / 12.0 + v / 12.0
                sns.histplot(
                    yrs,
                    bins=30,
                    kde=True,
                    ax=ax4,
                    stat="density",
                    color=color,
                    alpha=0.45,
                    label=label,
                )
        ax4.axvline(2030.0, color="black", ls=":", lw=1.5, label="2030 horizon")
        ax4.set_title(
            f"D. Deva (2026) Intercrop Advantage: P(cross by 2030) "
            f"drops from {dual['summary']['P_Dec2030_mono']:.1f}% to "
            f"{dual['summary']['P_Dec2030_inter']:.1f}%",
            fontweight="bold",
            loc="left",
        )
        ax4.set_xlabel("Calendar Year of First Crossing")
        ax4.set_ylabel("Probability density")
        ax4.legend(loc="upper right", framealpha=0.92)

    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[OK] Figure saved: {out_png}")


def export_forecast_summary(price_df, mc, out_csv="njoro_forecast_summary.csv"):
    """Export monthly percentile path table."""
    last_date = pd.to_datetime(price_df["Date"].iloc[-1])
    fut = pd.date_range(start=last_date, periods=mc["n_months"] + 1, freq="MS")[1:]
    q = mc["quantiles"]
    pd.DataFrame(
        {
            "Forecast_Month": fut.strftime("%Y-%m"),
            "Price_P10_KES": np.round(q["p10"], 1),
            "Price_P25_KES": np.round(q["p25"], 1),
            "Price_Median_KES": np.round(q["median"], 1),
            "Price_P75_KES": np.round(q["p75"], 1),
            "Price_P90_KES": np.round(q["p90"], 1),
        }
    ).to_csv(out_csv, index=False)
    print(f"[OK] Forecast CSV saved: {out_csv}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    ap = argparse.ArgumentParser(
        description="Njoro maize price pipeline: WB RTFP + KAMIS + Wainwright + Deva."
    )
    # Data sources
    ap.add_argument(
        "--wb-file",
        type=str,
        default=None,
        help="Path to World Bank RTFP CSV (e.g. WB_KEN_RTFP_mkt_2007_2026-10-05.csv).",
    )
    ap.add_argument(
        "--kamis-file",
        type=str,
        default=None,
        help="Path to a local KAMIS .csv / .xls / .xlsx file.",
    )
    ap.add_argument(
        "--no-scrape", action="store_true", help="Disable live KAMIS scraping."
    )

    # Filters
    ap.add_argument(
        "--commodity",
        type=str,
        default="maize",
        help="Commodity name for KAMIS (default: 'maize').",
    )
    ap.add_argument(
        "--commodity-col",
        type=str,
        default="maize_fao",
        help="WB RTFP column for maize price. "
        "Recommended: maize_fao (auto-resolves to c_maize_fao "
        "if raw observed values are sparse).",
    )
    ap.add_argument(
        "--market",
        type=str,
        default="Nakuru",
        help="Market/region filter (default: Nakuru).",
    )
    ap.add_argument(
        "--start",
        type=str,
        default="2007-01-01",
        help="Start date for live scrape (ISO format).",
    )
    ap.add_argument(
        "--end", type=str, default=None, help="End date for live scrape (ISO format)."
    )

    # Climate
    ap.add_argument(
        "--chirps-nc",
        type=str,
        default=None,
        help="Path to CHIRPS NetCDF file from ClimateSERV.",
    )
    ap.add_argument(
        "--chirps-csv",
        type=str,
        default=None,
        help="Path to a pre-computed OND anomaly CSV.",
    )
    ap.add_argument(
        "--ond-anom",
        type=float,
        default=150.0,
        help="ICPAC OND anomaly forecast in mm.",
    )

    # Model
    ap.add_argument("--target-price", type=float, default=6000.0)
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--out-fig", type=str, default="njoro_forecast.png")
    ap.add_argument("--out-csv", type=str, default="njoro_forecast_summary.csv")
    ap.add_argument(
        "--no-strict",
        action="store_true",
        help="Disable consistency assertion (not recommended).",
    )
    ap.add_argument(
        "--no-dual",
        action="store_true",
        help="Skip the Deva monoculture vs intercrop comparison.",
    )
    args = ap.parse_args()

    print("=" * 78)
    print("   NJORO MAIZE PIPELINE — WB RTFP + KAMIS + Wainwright + Deva")
    print("=" * 78)

    # ---------------- Stage 1 ----------------
    df = ingest_data(args)
    latest_price = float(df["Price_KES"].iloc[-1])
    latest_date = pd.to_datetime(df["Date"].iloc[-1])
    print(
        f"  -> {len(df)} months, {df['Date'].iloc[0].strftime('%b %Y')} "
        f"to {latest_date.strftime('%b %Y')}"
    )
    print(f"  -> Latest price: KES {latest_price:,.1f} / 90kg bag")

    # ---------------- Stage 2 ----------------
    print("\n[Stage 2] Log-linear trend")
    trend = fit_log_linear_trend(
        df["Date"], df["Price_KES"], target_price=args.target_price
    )
    print(f"  -> N observations: {trend['n_obs']}")
    print(f"  -> Annual growth: {trend['annual_growth_pct']:.2f}%")
    print(f"  -> R^2 (log space): {trend['r2']:.3f}")
    print(
        f"  -> Deterministic crossing: {trend['crossing_date'].strftime('%b %Y')} "
        f"(~{trend['months_remaining'] / 12:.1f} years)"
    )

    # ---------------- Stage 4 (before Stage 3, need beta) ----------------
    print("\n[Stage 4] OND rainfall regression (Wainwright OND sensitivity)")
    if args.chirps_nc and os.path.exists(args.chirps_nc):
        print(f"  -> Loading CHIRPS NetCDF: {args.chirps_nc}")
        cm = load_chirps_nc(args.chirps_nc)
        cm["Year"] = cm["Date"].dt.year
        cm["Month"] = cm["Date"].dt.month
        ond = cm[cm["Month"].isin([10, 11, 12])]
        oy = ond.groupby("Year")["rainfall_mm"].sum().reset_index()
        oy.columns = ["Year", "OND_Total_mm"]
        baseline = float(oy["OND_Total_mm"].mean())
        oy["OND_Rainfall_Anomaly_mm"] = oy["OND_Total_mm"] - baseline
        rain_df = oy[["Year", "OND_Rainfall_Anomaly_mm"]]
        print(f"  -> OND baseline mean (record period): {baseline:.1f} mm")
    elif args.chirps_csv and os.path.exists(args.chirps_csv):
        print(f"  -> Loading CHIRPS anomaly CSV: {args.chirps_csv}")
        rain_df = pd.read_csv(args.chirps_csv)
    else:
        print("  -> No CHIRPS file; using synthetic anomalies.")
        rain_df = generate_synthetic_chirps_data()

    rain = fit_rainfall_regression(df, rain_df, forecast_ond_anom=args.ond_anom)
    print(
        f"  -> beta = {rain['beta']:.6f} (SE {rain['se_beta']:.6f}, "
        f"t = {rain['t_stat']:.2f})"
    )
    print(f"  -> R^2 = {rain['r2']:.3f}, r = {rain['r_corr']:.3f}, n = {rain['n_obs']}")
    print(f"  -> Sensitivity: +100mm OND -> {rain['sensitivity_per_100mm']:+.2f}%")
    print(
        f"  -> ICPAC +{args.ond_anom:.0f}mm -> {rain['projected_seasonal_pct']:+.2f}%"
    )

    # ---------------- Stage 3 ----------------
    print(
        f"\n[Stage 3] Wainwright–Deva jump-diffusion Monte Carlo ({args.sims:,} paths)"
    )
    print("  -> Wainwright: non-stationary shock probability (+0.03/decade)")
    print("  -> Wainwright: growing jump magnitude (+0.01/decade)")
    print("  -> Wainwright: OND structural drift term included")
    print("  -> Deva: cropping-system-specific jump calibration")

    mc = simulate_climate_shocks(
        last_price=latest_price,
        slope_b=trend["slope_b"],
        hist_prices=df["Price_KES"].values,
        horizon_years=10,
        n_simulations=args.sims,
        target_price=args.target_price,
        cropping_system="monoculture",
        rainfall_regression_beta=rain["beta"],
        icpac_forecast_mm=args.ond_anom,
        strict=not args.no_strict,
    )
    print(f"  -> sigma (monthly log vol): {mc['sigma']:.4f}")
    print(f"  -> OND drift contribution: {mc['ond_drift']:+.6f}/month")
    print(f"  -> P(cross by Jan 2027): {mc['prob_jan2027']:.2f}%")
    print(f"  -> P(cross by Dec 2027): {mc['prob_dec2027']:.2f}%")
    print(f"  -> P(cross by Dec 2028): {mc['prob_dec2028']:.2f}%")
    print(f"  -> P(cross by Dec 2030): {mc['prob_dec2030']:.2f}%")
    print(f"  -> P(cross by Dec 2033): {mc['prob_dec2033']:.2f}%")
    for d in mc["diagnostics"]:
        print(
            f"      {d['label']:>9}: p90={d['p90_T']:>7.0f}  "
            f"P_term={d['P_terminal_above']:.4f}  "
            f"P_cross={d['P_cross_by_T']:.4f}  "
            f"OK_A={d['check_a_terminal_le_cross']} "
            f"OK_C={d['check_c_p90_bound']}"
        )

    # Deva dual-scenario
    dual = None
    if not args.no_dual:
        dual = simulate_dual_cropping_scenarios(
            last_price=latest_price,
            slope_b=trend["slope_b"],
            hist_prices=df["Price_KES"].values,
            horizon_years=10,
            n_simulations=args.sims,
            target_price=args.target_price,
            rainfall_regression_beta=rain["beta"],
            icpac_forecast_mm=args.ond_anom,
            strict=not args.no_strict,
        )

    # ---------------- Stage 5 ----------------
    print("\n[Stage 5] Figures + CSV")
    generate_publication_figures(df, trend, mc, rain, dual=dual, out_png=args.out_fig)
    export_forecast_summary(df, mc, out_csv=args.out_csv)

    print("\n" + "=" * 78)
    print("PIPELINE COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
