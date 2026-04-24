"""
regime_classifier.py
====================
Loads the trained models from disk and classifies the current 4H bar.

Outputs three things the strategies need:
  - gb_prob      : P(trending), from gb_model. Range 0.0-1.0
  - prob_ranging : rf3 class 0 probability (ranging). Range 0.0-1.0
  - trend_dir    : 'up', 'down', or None (from rf3 3-class model)

SOURCE: MeanReversion_Strategy.ipynb Cell 2 / Cell 2b
        Regime_Classifier_v2.ipynb Cell 5 (label structure)

HOW IT WORKS:
  gb_model is a binary GradientBoosting classifier.
    class 0 = ranging, class 1 = trending
    gb_prob = predict_proba()[:, 1]  <- probability of class 1 (trending)

  rf3 is a 3-class RandomForest classifier.
    class 0 = ranging
    class 1 = trend_up   (trending AND close > MA50)
    class 2 = trend_down (trending AND close < MA50)
    prob_ranging = rf3.predict_proba()[:, 0]  <- class 0 probability
    This matches the backtest all_data.parquet prob_ranging values exactly.

  NOTE: prob_ranging is NOT 1 - gb_prob.
  The backtest used rf3 class 0 probability for prob_ranging.
  Using 1 - gb_prob gives 2x more ranging bars and causes MR to overtrade.
"""

# Fix for numpy BitGenerator deserialization on some environments
import numpy.random
numpy.random.MT19937 = numpy.random._mt19937.MT19937

import numpy as np
import joblib
import pickle
import os
from regime_features import compute_regime_features, FEATURE_COLS


class RegimeClassifier:
    """
    Loads and runs the regime models on a 4H OHLCV DataFrame.

    Usage:
        clf = RegimeClassifier(models_dir='models/')
        result = clf.classify(df_4h)
        # result = {
        #   'gb_prob': 0.87,
        #   'prob_ranging': 0.13,
        #   'trend_dir': 'up'   or 'down' or None
        # }
    """

    def __init__(self, models_dir: str = 'models/'):
        """
        Load models from disk once at startup.
        Call this once when the bot starts -- not on every bar.
        """
        self.models_dir = models_dir
        self._load_models()

    def _load_models(self):
        """Load gb_model, rf3, and feature_cols from disk."""

        gb_path   = os.path.join(self.models_dir, 'gb_model.pkl')
        rf3_path  = os.path.join(self.models_dir, 'rf3_model.pkl')
        feat_path = os.path.join(self.models_dir, 'feature_cols.pkl')

        if not os.path.exists(gb_path):
            raise FileNotFoundError(
                f"gb_model.pkl not found at {gb_path}\n"
                "Copy your model files from Google Drive to the models/ folder."
            )

        self.gb_model = joblib.load(gb_path)
        self.rf3      = joblib.load(rf3_path)

        with open(feat_path, 'rb') as f:
            saved_cols = pickle.load(f)

        if saved_cols != FEATURE_COLS:
            print("WARNING: feature_cols.pkl differs from FEATURE_COLS in code.")
            print(f"   Saved  : {saved_cols}")
            print(f"   In code: {FEATURE_COLS}")
            print("   Using saved cols as ground truth.")
            self._feature_cols = saved_cols
        else:
            self._feature_cols = FEATURE_COLS

        self._rf3_classes = list(self.rf3.classes_)
        print(f"Models loaded")
        print(f"   gb_model  : {type(self.gb_model).__name__}")
        print(f"   rf3       : {type(self.rf3).__name__} | classes={self._rf3_classes}")
        print(f"   features  : {len(self._feature_cols)} columns")
        print(f"NOTE: In the parquet, prob_ranging != 1 - gb_prob.")
        print(f"      prob_ranging = rf3 class 0 probability (matches backtest exactly).")

    def classify(self, df_4h) -> dict:
        """
        Run regime classification on a 4H OHLCV DataFrame.

        Args:
            df_4h: DataFrame with columns Open, High, Low, Close, Volume
                   Index must be datetime, sorted ascending.
                   Needs at least 100 bars.

        Returns:
            dict with keys:
              gb_prob      (float)   : P(trending), 0.0-1.0
              prob_ranging (float)   : rf3 class 0 probability, 0.0-1.0
              trend_dir    (str|None): 'up', 'down', or None if not trending
              raw_rf3_pred (int)     : 0=ranging, 1=trend_up, 2=trend_down

        Returns None if not enough bars or features are NaN.
        """
        # Step 1: compute features
        feat_df = compute_regime_features(df_4h)
        last    = feat_df[self._feature_cols].iloc[-1]

        # Step 2: check for NaNs
        if last.isna().any():
            nan_cols = last[last.isna()].index.tolist()
            print(f"NaN in features: {nan_cols} -- skipping classification")
            return None

        # Step 3: reshape for sklearn
        X = last.values.reshape(1, -1)

        # Step 4: run gb_model (binary: ranging vs trending)
        gb_proba = self.gb_model.predict_proba(X)[0]
        gb_prob  = float(gb_proba[1])   # P(trending)

        # Step 5: run rf3 (3-class: 0=ranging, 1=trend_up, 2=trend_down)
        rf3_proba = self.rf3.predict_proba(X)[0]
        rf3_pred  = int(self.rf3.predict(X)[0])

        # prob_ranging = rf3 class 0 probability
        # This matches the backtest all_data.parquet exactly.
        # Do NOT use 1 - gb_prob -- that gives wrong regime distribution.
        prob_ranging = float(rf3_proba[0])

        # Translate rf3 prediction to direction
        if rf3_pred == 1:
            trend_dir = 'up'
        elif rf3_pred == 2:
            trend_dir = 'down'
        else:
            trend_dir = None

        return {
            'gb_prob':      round(gb_prob, 6),
            'prob_ranging': round(prob_ranging, 6),
            'trend_dir':    trend_dir,
            'raw_rf3_pred': rf3_pred,
        }

    def is_ranging(self, result: dict, threshold: float = 0.80) -> bool:
        """Returns True if MR strategy should be active."""
        if result is None:
            return False
        return result['prob_ranging'] > threshold

    def is_trending(self, result: dict, threshold: float = 0.90) -> tuple:
        """Returns (is_trending, direction) for the trend strategy."""
        if result is None:
            return False, None
        if result['gb_prob'] > threshold and result['trend_dir'] is not None:
            return True, result['trend_dir']
        return False, None


if __name__ == '__main__':
    import pandas as pd
    np.random.seed(0)
    n = 200
    close = 1.3000 + np.cumsum(np.random.randn(n) * 0.0008)
    df = pd.DataFrame({
        'Open':  close - abs(np.random.randn(n) * 0.0002),
        'High':  close + abs(np.random.randn(n) * 0.0004),
        'Low':   close - abs(np.random.randn(n) * 0.0004),
        'Close': close,
        'Volume': np.ones(n) * 1000,
    }, index=pd.date_range('2024-01-01', periods=n, freq='4h'))

    print("=== regime_classifier.py ===")
    print("Feature pipeline test:")
    from regime_features import compute_regime_features, FEATURE_COLS
    feats = compute_regime_features(df)
    last  = feats[FEATURE_COLS].iloc[-1]
    print(f"  Features computed: {last.notna().sum()} / 19 non-NaN")