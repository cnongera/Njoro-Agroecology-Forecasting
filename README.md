# Njoro Agroecology: Maize Price Volatility Pipeline

This repository contains a computational pipeline that models maize price volatility in Njoro Sub-County, Nakuru County, Kenya. It integrates real-world agricultural market data with state-of-the-art climate science and remote sensing research.

## Key Features
* **Real-World Data:** Ingests 238 months of Nakuru maize prices from the World Bank's RTFP database and 140 months of CHIRPS satellite rainfall data for the Njoro grid.
* **Wainwright Climate Calibration:** Models non-stationary climate shocks based on the documented shortening of the East African OND short-rains return period (Palmer, Wainwright et al., 2023).
* **Deva Intercrop Resilience:** Adjusts price shock magnitudes based on the Ensemble Stress Index (ENS) for monoculture vs. intercropped systems (Chakhar, Deva & Challinor, 2026).
* **Corrected Jump-Diffusion Monte Carlo:** Includes rigorous consistency checks to ensure that first-passage probabilities match empirical quantiles.

## Key Findings (Empirical)
* **Trend vs. Shock:** While a simple log-linear trend crossed KES 6,000/90kg bag in October 2022, the Monte Carlo model shows that under climate shocks, the probability of the price exceeding KES 6,000 is **46.9% by January 2027** and **94.3% by December 2030**.
* **The Rainfall Puzzle:** A linear regression of price changes on OND rainfall anomalies yields a negative coefficient (β = −0.00047, t = −0.87). A threshold regression at +50mm flips the sign (supporting Wainwright's mechanism), but with only 3 extreme-wet years in the sample, the result lacks statistical power.
* **Intercropping Benefit:** Under current high-price regimes, intercropping reduces the 2030 crossing probability by only 1.4 percentage points, because the base probability is already saturated.

## Repository Contents
* `njoro_pipeline.py`: The main executable pipeline (ingests data, runs the Monte Carlo, generates figures).
* `threshold_regression.py`: A script to run the threshold regression on the overlapping years.
* `Regime_Persistence_and_Climate_Sensitivity_in_Kenyan_Highland_Maize_Prices.md`: The full research paper.
* `Regime_Persistence_and_Climate_Sensitivity_in_Kenyan_Highland_Maize_Prices.pdf`: The typeset PDF version (generated via Kami(https://github.com/tw93/Kami)).
* `njoro_forecast.png`: The 4-panel publication-grade figure.
* `njoro_forecast_summary.csv`: The monthly p10–p90 price trajectory forecasts.

## Data Sources & Citations
* **Maize Prices:** World Bank Real-Time Food Prices (RTFP) database (Kenya, 2007–2026).
* **Rainfall:** CHIRPS 2.0 via ClimateSERV.
* **Climate Mechanics:** Palmer, P. I., Wainwright, C. M., et al. (2023). *Nature Reviews Earth & Environment*.
* **Agroecology Mechanics:** Chakhar, A., Deva, C., & Challinor, A. J. (2026). *Frontiers in Agronomy*.

## Usage
```bash
python3 njoro_pipeline.py \
    --wb-file KEN_RTFP_mkt_2007_2026-10-05.csv \
    --market Nakuru \
    --commodity-col maize_fao \
    --chirps-nc chirps_njoro.nc \
    --ond-anom 150 \
    --sims 5000
