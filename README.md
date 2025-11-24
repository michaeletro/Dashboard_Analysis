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

## Configuration
Environment variables let you customise data source, theme, layout sizing, and slider defaults. All optional; sensible defaults are provided.

| Variable | Purpose | Default |
|----------|---------|---------|
| `DATA_XLSX` | Path to Excel workbook (prices + equity data) | Fallback synthetic data if missing |
| `APP_TITLE` | Application title text | Bloomberg style portfolio dashboard |
| `APP_BG_COLOR` | Page background | `#000000` |
| `APP_PANEL_COLOR` | Plot background panels | `#111111` |
| `APP_CARD_COLOR` | Card background | `#151515` |
| `APP_ACCENT_COLOR` | Primary accent (titles/highlights) | `#f8e71c` |
| `APP_ACCENT2_COLOR` | Secondary accent | `#00e6ff` |
| `APP_GRID_COLOR` | Grid line color | `#333333` |
| `APP_TEXT_COLOR` | Primary text color | `#f5f5f5` |
| `APP_MUTED_COLOR` | Muted / secondary text | `#999999` |
| `APP_BORDER_COLOR` | Border and divider color | `#2a2a2a` |
| `CARD_PADDING` | Card CSS padding | `16px 18px` |
| `CARD_MIN_HEIGHT` | Card min-height CSS | `460px` |
| `FRONTIER_MIN` / `MAX` / `STEP` / `DEFAULT` | Efficient frontier slider bounds | `10 / 200 / 5 / 80` |
| `SIM_STEPS_MIN` / `MAX` / `STEP` / `DEFAULT` | GBM steps slider | `10 / 500 / 10 / 252` |
| `SIM_NPATHS_MIN` / `MAX` / `STEP` / `DEFAULT` | Number of paths | `10 / 1000 / 10 / 200` |
| `SIM_MAXPATHS_MIN` / `MAX` / `STEP` / `DEFAULT` | Max displayed paths | `10 / 200 / 10 / 50` |

Example (bash):
```bash
export DATA_XLSX="/data/my_universe.xlsx"
export APP_TITLE="My Quant Dashboard"
export APP_ACCENT_COLOR="#ff9900"
export FRONTIER_DEFAULT=120
export SIM_STEPS_DEFAULT=300
python app.py
```

At runtime you can inspect merged settings via:
```python
from dashboard.config import get_runtime_settings
settings = get_runtime_settings()
print(settings["APP_TITLE"], settings["FRONTIER"]["default"])
```

If you change environment variables, restart the app (current implementation reads them on import / first settings call).

## Next Upgrades
Already implemented:
* Lazy-load heavy data prep with synthetic fallback
* Centralised metrics (NMISE/MAPE) in `common/metrics.py`
* Environment-driven configuration + runtime settings accessor

Planned:
* Basic tests and GitHub Actions CI
* Settings panel in UI for live refresh of theme/slider defaults
* File picker for Excel upload and status indicators for long computations
