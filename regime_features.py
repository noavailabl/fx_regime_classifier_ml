"""
regime_features.py
==================
Computes the exact 19 regime features used to train gb_model and rf3.

SOURCE: Regime_Classifier_v2.ipynb — Cell 3 (indicator functions)
        and Cell 5 (build_features_and_labels).

CRITICAL RULE: Every formula here must match the training code exactly.
If even one feature is computed differently, gb_prob will be wrong
and all signal routing breaks. Do not change anything without retraining.

The 19 features in order (must match feature_cols.pkl):
  adx, adx_slope, adx_accel, di_spread,
  bb_width, bb_slope,
  hurst, hurst_slope,
  er_10, er_20, er_slope,
  ar1, adf_pval, ou_halflife,
  vol_ratio, atr_slope,
  price_vs_ma,
  dir_consist, ret_vol
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# ─────────────────────────────────────────────
# LOW-LEVEL INDICATOR FUNCTIONS
# These match Cell 3 of Regime_Classifier_v2
# ─────────────────────────────────────────────

def calc_atr(data, period=14):
    """
    Average True Range.
    True Range = max of:
      - High - Low
      - |High - prev Close|
      - |Low  - prev Close|
    Then rolling mean of TR.
    Returns: (raw TR series, smoothed ATR series)
    """
    tr1 = data['High'] - data['Low']
    tr2 = (data['High'] - data['Close'].shift()).abs()
    tr3 = (data['Low']  - data['Close'].shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr, tr.rolling(period).mean()


def calc_adx(data, tr, period=14):
    """
    ADX (Average Directional Index) + DI+ and DI-.
    Measures trend STRENGTH (not direction).
    High ADX = strong trend. Low ADX = ranging.

    Also returns plus_di and minus_di which tell direction.
    """
    up   = data['High'].diff()
    down = -data['Low'].diff()

    # Plus directional movement: up move that exceeded down move
    plus_dm  = pd.Series(np.where((up > down) & (up > 0), up, 0),   index=data.index)
    # Minus directional movement: down move that exceeded up move
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=data.index)

    tr14       = tr.rolling(period).sum()
    plus_di    = 100 * (plus_dm.rolling(period).sum()  / tr14)
    minus_di   = 100 * (minus_dm.rolling(period).sum() / tr14)
    dx         = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx        = dx.rolling(period).mean()

    return adx, plus_di, minus_di


def calc_hurst(series):
    """
    Hurst Exponent (rolling window applied externally).
    H > 0.5  → trending (persistent)
    H = 0.5  → random walk
    H < 0.5  → mean-reverting

    EXACT training code from Regime_Classifier_v2 Cell 7:
    - lags = range(2, 20)   ← NOT 28
    - uses series.diff(lag).dropna()  ← NOT np.diff(series, lag)
    Both matter for matching training output.
    """
    try:
        lags = range(2, 20)
        tau  = [np.std(series.diff(lag).dropna()) for lag in lags]
        if any(t == 0 for t in tau):
            return np.nan
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except:
        return np.nan


def calc_efficiency_ratio(close, period=14):
    """
    Efficiency Ratio (ER) — Perry Kaufman.
    ER = net displacement / total path
    ER → 1.0: price moved in one direction (trending)
    ER → 0.0: price zigzagged (ranging/noisy)
    """
    net  = abs(close - close.shift(period))
    path = close.diff().abs().rolling(period).sum()
    return net / path.replace(0, np.nan)


def calc_ari(series):
    """
    AR(1) coefficient (autocorrelation at lag 1).
    Positive/high → trending (momentum)
    Negative → mean-reverting
    Computed via np.corrcoef(y[:-1], y[1:]).
    """
    try:
        y = series.values
        return np.corrcoef(y[:-1], y[1:])[0, 1]
    except:
        return np.nan


def calc_adf_pval(series):
    """
    Augmented Dickey-Fuller p-value.
    Low p-value  → stationary → ranging.
    High p-value → unit root  → trending.
    maxlags=1, autolag=None (matches training exactly).

    IMPORTANT: This should never fail on real 4H MT5 data.
    Real FX bars always have enough variance for ADF to run.
    If it raises in live trading, the data feed has a problem.
    """
    try:
        return adfuller(series, maxlag=1, autolag=None)[1]
    except Exception as e:
        raise RuntimeError(f"adf_pval computation failed: {e}")


def calc_ou_halflife(series):
    """
    Ornstein-Uhlenbeck half-life.
    Short half-life → fast mean reversion → ranging.
    Long half-life → slow/no mean reversion → trending.
    Clipped to [0, 200] in feature engineering (matches training).
    """
    try:
        delta = np.diff(series)
        lag   = series[:-1]
        beta  = np.polyfit(lag, delta, 1)[0]
        if beta >= 0:
            return 999.0
        return min(-np.log(2) / beta, 999.0)
    except:
        return np.nan


# ─────────────────────────────────────────────
# MAIN FEATURE BUILDER
# Matches Cell 5 of Regime_Classifier_v2
# Input:  DataFrame with OHLCV on 4H timeframe
#         needs at least 100 bars for rolling windows
# Output: DataFrame with 19 feature columns
# ─────────────────────────────────────────────

def compute_regime_features(d: pd.DataFrame) -> pd.DataFrame:
    """
    Given a 4H OHLCV DataFrame, compute all 19 regime features.

    Column names must be: Open, High, Low, Close, Volume (or tick_volume)
    Index must be datetime.

    Returns a DataFrame with the same index and exactly 19 feature columns.
    The last row is the current (most recent) bar.
    """
    d = d.copy()

    # ── Base indicators ──────────────────────────────────────────
    tr, d['atr'] = calc_atr(d)
    d['adx'], d['plus_di'], d['minus_di'] = calc_adx(d, tr)

    # Bollinger Band width (not pct — just width relative to price)
    bb_ma  = d['Close'].rolling(20).mean()
    bb_std = d['Close'].rolling(20).std()
    d['bb_width'] = (4 * bb_std) / d['Close']   # 2σ each side = 4σ total / price

    # 50-period MA for price_vs_ma feature
    d['ma_50'] = d['Close'].rolling(50).mean()

    # Hurst — rolling 40-bar window, applied bar by bar
    # raw=False: passes a Series not an array (needed for calc_hurst)
    d['hurst'] = d['Close'].rolling(40).apply(calc_hurst, raw=False)

    # ── Feature DataFrame ─────────────────────────────────────────
    feat = pd.DataFrame(index=d.index)

    # ADX family (trend strength)
    feat['adx']       = d['adx']
    feat['adx_slope'] = d['adx'].diff(3)        # rate of change over 3 bars
    feat['adx_accel'] = d['adx'].diff(3).diff(3) # acceleration of ADX
    feat['di_spread'] = abs(d['plus_di'] - d['minus_di'])  # DI+ minus DI-

    # Bollinger Band family (volatility regime)
    feat['bb_width'] = d['bb_width']
    feat['bb_slope'] = d['bb_width'].diff(3)    # expanding or contracting bands?

    # Hurst family (memory / persistence)
    feat['hurst']       = d['hurst']
    feat['hurst_slope'] = d['hurst'].diff(3)

    # Efficiency Ratio family (directional efficiency)
    feat['er_10']    = calc_efficiency_ratio(d['Close'], 10)
    feat['er_20']    = calc_efficiency_ratio(d['Close'], 20)
    feat['er_slope'] = feat['er_10'].diff(3)

    # AR(1) — autocorrelation (rolling 20 bars)
    feat['ar1'] = d['Close'].rolling(20).apply(calc_ari, raw=False)

    # ADF p-value — stationarity test (rolling 30 bars)
    feat['adf_pval'] = d['Close'].rolling(30).apply(calc_adf_pval, raw=False)

    # OU half-life — mean reversion speed (rolling 32 bars, clipped 0-200)
    feat['ou_halflife'] = d['Close'].rolling(30).apply(
        calc_ou_halflife, raw=False
    ).clip(0, 200)

    # Volatility regime
    feat['vol_ratio']  = d['atr'] / d['atr'].rolling(50).mean()   # current vs avg ATR
    feat['atr_slope']  = d['atr'].diff(3)

    # Price position relative to trend
    feat['price_vs_ma'] = (d['Close'] - d['ma_50']) / d['atr']

    # Directional consistency (are candles mostly bullish or mostly bearish?)
    direction = (d['Close'] > d['Open']).astype(int) * 2 - 1  # +1 bull, -1 bear
    feat['dir_consist'] = direction.rolling(5).sum().abs()    # 0=mixed, 5=all same

    # Return volatility (how noisy is price?)
    feat['ret_vol'] = d['Close'].pct_change().rolling(10).std()

    return feat


# ─────────────────────────────────────────────
# FEATURE COLUMN ORDER
# Must match feature_cols.pkl exactly.
# Used when calling model.predict_proba(feat[FEATURE_COLS])
# ─────────────────────────────────────────────

FEATURE_COLS = [
    'adx', 'adx_slope', 'adx_accel', 'di_spread',
    'bb_width', 'bb_slope',
    'hurst', 'hurst_slope',
    'er_10', 'er_20', 'er_slope',
    'ar1', 'adf_pval', 'ou_halflife',
    'vol_ratio', 'atr_slope',
    'price_vs_ma',
    'dir_consist', 'ret_vol',
]
# 19 features total — verified against notebook output


if __name__ == '__main__':
    # Quick sanity check — generates fake OHLCV and prints feature output
    np.random.seed(42)
    n = 200
    close = 1.1000 + np.cumsum(np.random.randn(n) * 0.0005)
    df = pd.DataFrame({
        'Open':  close - abs(np.random.randn(n) * 0.0002),
        'High':  close + abs(np.random.randn(n) * 0.0003),
        'Low':   close - abs(np.random.randn(n) * 0.0003),
        'Close': close,
        'Volume': np.random.randint(100, 1000, n).astype(float),
    }, index=pd.date_range('2024-01-01', periods=n, freq='4h'))

    feats = compute_regime_features(df)
    last  = feats[FEATURE_COLS].iloc[-1]

    print("=== regime_features.py sanity check ===")
    print(f"Total features : {len(FEATURE_COLS)}")
    print(f"Non-NaN on last bar: {last.notna().sum()} / {len(FEATURE_COLS)}")
    print()
    print("Last bar values:")
    for col in FEATURE_COLS:
        print(f"  {col:<20}: {last[col]:.6f}")