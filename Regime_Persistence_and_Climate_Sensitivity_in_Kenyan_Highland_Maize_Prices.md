# Regime Persistence and Climate Sensitivity in Kenyan Highland Maize Prices: A Jump-Diffusion Analysis of Nakuru Market Data, 2007–2026

**Author:** Clement Ongera Nyangoya
**Affiliation:** Independent researcher, Nakuru County, Kenya
**Date:** October 2026
**Status:** Working paper (submitted for discussion, not peer-reviewed)
**Keywords:** Agroecology, maize price volatility, climate teleconnections, ENSO, jump-diffusion, Wainwright rainfall mechanics, Deva intercrop advantage, Kenya.

---

## Abstract

Smallholder maize systems in the Kenyan highlands are exposed to compound volatility from two sources: large-scale oceanic teleconnections (ENSO and the Indian Ocean Dipole) that intensify the October–December short rains, and market dynamics that amplify local price shocks. We present a **jump-diffusion Monte Carlo pipeline** calibrated to two recent bodies of empirical work — Palmer, Wainwright et al. (2023) on East African rainfall variability, and Chakhar, Deva & Challinor (2026) on maize intercrop resilience — and apply it to **real data**:

- **238 months** of Nakuru maize wholesale prices from the World Bank Real-Time Food Prices (RTFP) database, January 2007 – October 2026.
- **140 months** of OND rainfall at the Njoro grid cell from CHIRPS 2.0 via ClimateSERV.

Three findings emerge.

First, the deterministic log-linear trend fit on the full 238-month series crosses KES 6,000 per 90 kg bag in **October 2022** — a historical event, not a forecast. The current model-central price is KES 5,632, so the trend line is above the current spot price. This reflects a level shift from the 2022–2023 drought and El Niño harvest losses, and it demonstrates that simple trend extrapolation is inappropriate for regime-shifted series.

Second, a linear regression of post-harvest price change on OND rainfall anomaly returns a **negative and statistically insignificant coefficient** (β = −0.000473, SE = 0.000542, t = −0.87, n = 11). The naive reading of the Wainwright mechanism predicts a positive coefficient. We interpret the negative sign as evidence that (i) in most years, wet short-rains seasons increase supply and depress prices; (ii) the Wainwright mechanism operates only above a threshold of extreme wetness and requires a non-linear specification to detect. With 11 overlapping years, we lack the statistical power to test this rigorously; we present it as a puzzle for future work.

Third, under the calibrated Monte Carlo, the probability that Nakuru prices exceed KES 6,000 is **46.9% by January 2027**, **79.7% by December 2027**, and **94.3% by December 2030**. The Deva intercrop calibration reduces the 2030 crossing probability by only **1.4 percentage points** (from 94.3% to 92.9%), because the base probability is saturated near 100%. Agroecological interventions matter most in years when prices are not already dislocated.

We are explicit about limitations: the World Bank `c_maize_fao` column is a model central estimate, not raw survey data; the rainfall regression has only 11 observations; the deterministic trend crossing in the past is a diagnostic, not a forecast.

---

## 1. Introduction

Maize provides over 65% of daily caloric intake in Kenya and dominates smallholder acreage in Nakuru County, particularly in the highland sub-counties of Njoro, Rongai, and Molo (1,800–2,400 m ASL). Despite high agronomic potential, the maize production-consumption nexus in these areas is vulnerable to climate extremes and pricing instability. In late 2026, wholesale depot prices in Njoro village fluctuate around KES 5,000–5,600 per 90 kg bag, while urban aggregation centres in Nakuru Town command a small premium.

Two streams of recent research are directly relevant to this system:

1. **Wainwright and colleagues** (Palmer, Wainwright et al., 2023; Wainwright et al., 2020) established that the OND short rains over East Africa have been trending wetter since the mid-1980s; that extreme wet years cause harvest-time deluges, crop rotting, and post-harvest losses; and that the return period of extreme events has shortened from roughly 10 years (pre-1980) to 3.0–3.5 years since 2005.

2. **Deva and colleagues** (Chakhar, Deva & Challinor, 2026; Challinor, Deva et al., 2018) used Sentinel-1 SAR and Sentinel-2 multispectral imagery to quantify vegetation stress in maize systems across western Kenya. They showed that monoculture maize suffers severe unbuffered stress under moisture extremes, while intercropped polycultures maintain stable canopies — a difference they quantify with an **Intercrop Advantage Score (IAS)** and **Ensemble Stress Index (ENS)**.

This paper operationalises both bodies of work in a computational pipeline and applies it to real Nakuru price and Njoro rainfall data. Our contributions are threefold:

1. A **corrected jump-diffusion Monte Carlo** with explicit consistency checks (Section 3.3).
2. **Non-stationary climate-shock parameters** calibrated to Wainwright's empirically documented return-period shortening (Section 3.3.2).
3. **Cropping-system scenarios** calibrated to Deva's ENS evidence, allowing a quantitative comparison of monoculture vs. intercropped price-risk exposure (Section 3.3.3).

We do not claim any of these components are individually novel. What is new is the integration and application to real data, and the honest reporting of three results that complicate the naive reading of both bodies of work.

---

## 2. Data

### 2.1 Maize prices

We use the World Bank's **Real-Time Food Prices (RTFP)** dataset, version 2026-10-05, for Kenya. This dataset provides monthly food price estimates for 233 Kenyan markets from January 2007 to the present, produced using a machine-learning imputation model (Cubist; see the World Bank metadata file accompanying the dataset).

We extract:

- Market filter: Nakuru (matched against `adm1_name`, `adm2_name`, or `mkt_name`).
- Commodity column: `c_maize_fao` — the model's **central estimate** for maize prices. The raw observed column `maize_fao` is sparse (3,045 non-null rows out of 55,692), so we use the central estimate for continuous monthly coverage. This is standard practice for RTFP-based analyses.
- Currency: KES per kilogram, converted to **KES per 90 kg bag** by multiplying by 90.

The resulting series has **238 monthly observations**, from January 2007 to October 2026. The latest value is **KES 5,632.2 per 90 kg bag**.

**Caveat.** The central estimates are model outputs, not raw survey values. The metadata file reports an imputation R² of 0.97 for maize. We treat the series as a high-quality estimate but not ground truth.

### 2.2 Rainfall

We use **CHIRPS 2.0** (Climate Hazards Group InfraRed Precipitation with Stations) daily rainfall, downloaded via the ClimateSERV portal for a bounding box around Njoro Sub-County (approximately 0.33° S, 35.94° E). The nearest grid cell to the target location is (−0.325°, 35.925°), matching the Njoro highlands. The file spans January 2015 to October 2026, giving **140 months** of daily precipitation.

Daily values are aggregated to monthly totals. The OND anomaly for each year is the sum of October + November + December rainfall minus the sample-period mean. We use the sample mean rather than the 1981–2010 climatological baseline because the CHIRPS file we downloaded does not include the full baseline period.

The OND baseline mean (record period) is **272.3 mm**.

### 2.3 Overlap and sample size

The price series spans 2007–2026 and the rainfall series spans 2015–2026. The overlapping years for the rainfall regression are **2015–2026, giving n = 11**. This is thin and limits statistical power in Section 4.3.

---

## 3. Methods

### 3.1 Log-linear trend

We fit a log-linear trend to the full 238-month price series:
