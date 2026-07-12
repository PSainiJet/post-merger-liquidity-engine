# Global Intraday Liquidity Predictor & Stress Tester

A standalone AI system that predicts intraday liquidity needs for a post-merger global bank, flags currency-specific shortfall risk within a 48-hour horizon, and stress-tests bank-run scenarios.

> **Context:** When Credit Suisse failed in 2023, the root cause was a liquidity crisis — not solvency. Following UBS's acquisition and the March 2026 migration of Swiss client accounts, the combined institution still manages overlapping global IT infrastructures and elevated daily outflows. This project simulates that consolidation challenge.

## Problem Statement

Traditional treasury systems track *current* cash positions. This engine goes further:

1. **Predicts** how much liquid cash the merged bank needs on any given day
2. **Simulates** sudden market shocks and bank-run withdrawal patterns
3. **Flags** which currencies (CHF, USD, EUR) are at risk of a liquidity shortfall in the next 48 hours

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Simulated Data Streams                        │
├──────────────┬──────────────┬───────────────┬───────────────────┤
│ CS Legacy    │ UBS Core     │ Interbank     │ Corporate         │
│ Positions    │ Positions    │ Lending Rates │ Withdrawals       │
├──────────────┴──────────────┴───────────────┴───────────────────┤
│                    Derivative Margin Calls                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Feature Pipeline │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │     Ensemble Model          │
              ├─────────────┬───────────────┤
              │ Random      │ ARIMA Time    │
              │ Forest      │ Series        │
              │ (45%)       │ (55%)         │
              └──────┬──────┴───────┬───────┘
                     │              │
              ┌──────▼──────────────▼───────┐
              │  48h Liquidity Forecast     │
              │  + Stress Test Scenarios    │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  Treasury Dashboard         │
              │  CHF / USD / EUR Alerts     │
              └─────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- pip

### Setup

```bash
cd global-intraday-liquidity-predictor

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -e ".[dev]"
```

### Generate Simulated Data

```bash
python scripts/generate_simulated_data.py
```

This creates parquet files in `data/simulated/` from:
- Legacy Credit Suisse system feeds
- UBS core treasury positions
- Interbank lending rates (CHF, USD, EUR)
- Corporate withdrawal histories by client segment
- Derivative margin call obligations

### Train the Ensemble Model

```bash
python scripts/train_model.py
```

Trains Random Forest + ARIMA time-series models and saves artifacts to `models/artifacts/`.

### Launch the Treasury Dashboard

```bash
streamlit run src/dashboard/app.py
```

The dashboard shows:
- **Alert banners** for currencies at WARNING or CRITICAL risk
- **48-hour liquidity forecasts** per currency
- **Ensemble breakdown** (RF vs time-series components)
- **Recommended actions** (borrow, liquidate HQLA, monitor)
- **Stress test results** for negative news, market shock, and idiosyncratic run scenarios

## Project Structure

```
global-intraday-liquidity-predictor/
├── config/settings.yaml          # Thresholds, model weights, stress scenarios
├── src/
│   ├── data_simulation/          # Simulated CS/UBS feeds and market data
│   ├── models/                   # Random Forest + ARIMA ensemble
│   ├── stress_testing/           # Bank-run scenario engine
│   ├── prediction/               # Liquidity report generator
│   └── dashboard/                # Streamlit Treasury desk UI
├── scripts/                      # CLI entry points
├── tests/                        # Pipeline tests
└── data/simulated/               # Generated parquet files (gitignored)
```

## Stress Scenarios

| Scenario            | Withdrawal Multiplier | Margin Call Spike | Spread Widening |
|---------------------|----------------------|-------------------|-----------------|
| Negative News       | 2.5x                 | +40%              | +75 bps         |
| Market Shock        | 4.0x                 | +80%              | +150 bps        |
| Idiosyncratic Run   | 6.0x                 | +120%             | +200 bps        |

## Configuration

Edit `config/settings.yaml` to adjust:
- Liquidity thresholds per currency
- Model ensemble weights
- Stress scenario severity
- Simulation history length

## Running Tests

```bash
pytest
```

## Roadmap

- [ ] Real-time feed connectors (Kafka/SWIFT simulation)
- [ ] LCR/NSFR regulatory ratio integration
- [ ] Multi-currency FX conversion layer
- [ ] Historical calibration against public bank-run events
- [ ] REST API for treasury system integration
- [ ] Alerting webhooks (Slack/Teams/PagerDuty)

## Disclaimer

This is a research and simulation project. All data is synthetically generated. It is not affiliated with UBS, Credit Suisse, or any financial institution.

## License

MIT
=======
