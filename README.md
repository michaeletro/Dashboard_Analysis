# Bloomberg Style Portfolio Dashboard

A Dash application for portfolio optimisation (GBM-based) and equity dataframe analysis (factors, clustering, regression, peer analysis).

## Quick Start

```bash
# Python 3.10+
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Optional: point to your Excel workbook
# Defaults to ../DBG Data Set Presentation Prep Doc.xlsx relative to dashboard/data.py
export DATA_XLSX="/absolute/path/to/your.xlsx"

# Run the app
python app.py  # http://127.0.0.1:8051
```

If no Excel is found, the app falls back to synthetic demo data for both the time series engine and the equity cross section.

## Data Expectations

- Time series sheet: price levels (rows = dates, columns = asset tickers)
- Equity sheet: fundamentals and returns with columns like `1M % Change`, valuation ratios, profitability, market cap, etc.

## Repo Structure (key parts)
- `dashboard/`: Dash app wiring (layout, callbacks, figures, theme)
- `equity_data_frame_module/`: Factor construction, clustering, regression, peer analysis, portfolio utilities
- `time_series_modules/`: GBM calibration, frontier, and portfolio simulation engine

## Key Features

* Portfolio optimization using Geometric Brownian Motion (GBM) models
* Efficient frontier visualization with customizable parameters
* Equity analysis including factor construction, clustering, and regression
* Peer analysis and portfolio utilities
* Bloomberg-inspired dark theme with professional styling
* Synthetic data fallback for demo purposes when no data file is provided
