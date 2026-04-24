# FX Regime Classifier
### ML-Based Market Regime Detection for Algorithmic FX Trading

> *"I stopped trying to predict price. I started predicting what kind of market I was in."*

---

## Overview

Most trading strategies fail not because the entry logic is wrong — but because they run in the wrong market condition. A mean reversion strategy in a trending market doesn't just underperform. It systematically loses because the core assumption is violated.

This project builds a **regime classifier** that answers one question before any trade fires:

**Is this the right environment for my strategy right now?**

The classifier is trained on 204,230 samples across 6 FX pairs spanning 21 years (2005–2026) and deployed live on a MetaTrader 5 prop firm challenge account.

---

## Architecture

Two independent ML models form a dual-confirmation gate:

| Model | Type | Output | Purpose |
|-------|------|--------|---------|
| **Gradient Boosting** | Binary classifier | `gb_prob` = P(trending) | Detects trending regimes |
| **Random Forest (RF3)** | 3-class classifier | `prob_ranging`, `prob_trend_up`, `prob_trend_down` | Detects ranging + direction |

**Key design decision:** These models are not complements. They don't sum to 1. They're two independent models measuring the same underlying condition from different angles.

A mean reversion trade fires only when both agree:
- RF3: `prob_ranging > 0.80`
- GB: `gb_prob < threshold` (implicitly — low trending probability)

---

## Features (19 total, 4H timeframe)

| Category | Features |
|----------|---------|
| Trend strength | `adx`, `adx_slope`, `adx_accel`, `di_spread` |
| Volatility | `bb_width`, `bb_slope`, `vol_ratio`, `atr_slope` |
| Persistence | `hurst`, `hurst_slope`, `er_10`, `er_20`, `er_slope` |
| Mean reversion | `ar1`, `adf_pval`, `ou_halflife` |
| Price structure | `price_vs_ma`, `dir_consist`, `ret_vol` |

Notable: `adf_pval` (ADF stationarity test) and `ou_halflife` (Ornstein-Uhlenbeck half-life) were **not** in the labeling conditions but ranked in the top 5 features in both models. Independent corroborating evidence.

---

## Labeling Methodology

**Important:** Labels are constructed from current-bar information only. No future data.

```python
# Trending = at least 3 of 4 current indicators agree
cond_adx   = (adx > 25)
cond_bb    = (bb_width > bb_width.rolling(20).mean())
cond_hurst = (hurst > 0.55)
cond_er    = (er_10 > er_10.rolling(20).mean())

trending = (cond_adx + cond_bb + cond_hurst + cond_er) >= 3
```

The first labeling approach used KMeans clustering on **forward** efficiency ratio (`shift(-10)`). This introduced look-ahead bias. It was caught and replaced with the current-bar voting ensemble above. The original code is preserved (commented out) in the notebook as documentation of what not to do.

---

## Results

### Model Performance

| Model | Train Acc | Test Acc | Test AUC |
|-------|-----------|----------|----------|
| Gradient Boosting | 92.9% | 92.4% | 0.981 |
| Random Forest | 89.8% | 89.7% | 0.964 |

AUC is consistent across all 6 pairs (0.980–0.984), indicating the classifier generalises across currency pairs rather than overfitting to one.

### Indicator Separation

| Indicator | Trending | Ranging | Separation |
|-----------|----------|---------|------------|
| Efficiency Ratio | 0.548 | 0.266 | +105.5% |
| Hurst Exponent | 0.694 | 0.373 | +85.9% |
| BB Width | 0.022 | 0.013 | +66.1% |
| ADX | 45.3 | 33.5 | +35.1% |

### Regime Persistence

Both regimes persist for ~20 hours on average across all pairs — answering the common criticism that regime detection is always lagging.

| Regime | Same next candle | Avg duration |
|--------|-----------------|--------------|
| Trending | ~91% | ~20 hours |
| Ranging | ~89% | ~19 hours |

### Threshold Decision

At `prob_ranging > 0.80` (live threshold):
- **97%+ precision** — when the system flags ranging, it's right 97% of the time
- **~25% candle coverage** — the system sits out 75% of the time
- Selective entries, high confidence only

---

## Emergent Behavior

During the April 2026 tariff shock — one of the most volatile USD periods in recent memory — both models simultaneously dropped into uncertainty. Neither crossed its threshold. Zero trades fired across all pairs for hours.

No "don't trade during news" rule was programmed. The system handled it organically because the features measuring market structure become ambiguous when structure itself breaks down.

---

## Honest Limitations

- Labeling thresholds (ADX > 25, Hurst > 0.55) are theoretically grounded but not sensitivity-tested
- The model partially learns to replicate its own label rules due to feature-label overlap — mitigated by the presence of independent features (ADF, OU half-life) in the top rankings
- Live performance data is early-stage — the system has been running since March 2026 and is still in execution debugging

---

## Repository Structure

```
fx-regime-classifier/
├── README.md
├── Regime_Classifier.ipynb      # Main research notebook
├── regime_features.py           # Feature computation pipeline
├── regime_classifier.py         # Classifier class for live deployment
├── models/
│   ├── gb_model.pkl             # Trained GB binary model
│   ├── rf3_model.pkl            # Trained RF 3-class model
│   └── feature_cols.pkl         # Feature column order
├── charts/
│   ├── regime_overlay_GBPUSD_2021.png
│   ├── feature_importance.png
│   ├── indicator_separation.png
│   ├── probability_distribution.png
│   └── precision_coverage.png
└── requirements.txt
```

---

## Requirements

```
pandas
numpy
scikit-learn
statsmodels
matplotlib
joblib
```

---

## Related

- LinkedIn article: [I stopped trying to predict price](https://www.linkedin.com/pulse/) ← add link after publishing
- Live deployment: MetaTrader 5, The5ers prop firm challenge
- Strategy: Multi-pair FX mean reversion (GBPUSD, USDCAD, USDCHF) gated by this classifier

---

## Author

**Zeel Kadvani**
Dual-degree graduate (B.Tech CS + MBA Finance), Nirma University
Building regime-conditional algorithmic trading systems

[LinkedIn](https://linkedin.com/in/zeel-kadvani) · [GitHub](https://github.com/noavailabl)
