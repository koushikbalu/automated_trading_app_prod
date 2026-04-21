"""Vectorised indicator computations and helper functions.

Ported from momentum_trading_backtest.py -- ATR, DMAs, inverse-vol
weighting, momentum return helpers, and performance metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def sma(series: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    return series.rolling(window).mean()


def ema(series: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    return series.ewm(span=window, adjust=False).mean()


# ---------------------------------------------------------------------------
# ATR (vectorised across all columns)
# ---------------------------------------------------------------------------

def compute_atr_df(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    return tr.rolling(window).mean()


def compute_atr_series(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


# ---------------------------------------------------------------------------
# Volatility & liquidity
# ---------------------------------------------------------------------------

def rolling_volatility(
    daily_returns: pd.Series | pd.DataFrame,
    window: int = 60,
) -> pd.Series | pd.DataFrame:
    return daily_returns.rolling(window).std(ddof=1) * np.sqrt(252)


def adv_126(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """126-day average daily traded value (price * volume)."""
    return (close * volume).rolling(126).mean()


# ---------------------------------------------------------------------------
# Inverse-volatility weighting with iterative cap redistribution
# ---------------------------------------------------------------------------

def capped_inverse_vol_weights(
    vol_series: pd.Series,
    max_weight: float,
) -> pd.Series:
    inv_vol = 1 / vol_series.replace(0, np.nan)
    w = inv_vol / inv_vol.sum()
    w = w.clip(upper=max_weight)

    if w.sum() == 0 or w.isna().all():
        return pd.Series(0.0, index=vol_series.index)

    w = w / w.sum()

    for _ in range(10):
        over = w > max_weight
        if not over.any():
            break
        w[over] = max_weight
        under = ~over
        residual = 1 - w[over].sum()
        if under.sum() == 0:
            break
        base = 1 / vol_series[under].replace(0, np.nan)
        if base.sum() == 0 or base.isna().all():
            break
        w.loc[under] = residual * base / base.sum()

    w = w.fillna(0)
    if w.sum() > 0:
        w = w / w.sum()
    return w


def blended_weights(
    vol_series: pd.Series,
    score_series: pd.Series,
    max_weight: float,
    score_blend: float = 0.4,
) -> pd.Series:
    """Blend inverse-vol weights with score-rank weights.

    ``score_blend=0.4`` means 40 % from momentum rank, 60 % from inverse-vol.
    Stocks with the strongest momentum scores get proportionally more capital
    instead of relying purely on low volatility.
    """
    inv_vol_w = capped_inverse_vol_weights(vol_series, max_weight)

    score_rank = score_series.rank(ascending=False)
    rank_w = 1 / score_rank
    rank_w = rank_w / rank_w.sum()

    blended = (1 - score_blend) * inv_vol_w + score_blend * rank_w
    blended = blended.clip(upper=max_weight)
    total = blended.sum()
    if total > 0:
        blended = blended / total
    return blended


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1
    return float(dd.min())


def annualized_return(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return float("nan")
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1)


def annualized_vol(daily_returns: pd.Series) -> float:
    return float(daily_returns.std(ddof=1) * np.sqrt(252))


def sharpe_ratio(daily_returns: pd.Series, annual_rf: float = 0.04) -> float:
    daily_rf = (1 + annual_rf) ** (1 / 252) - 1
    excess = daily_returns - daily_rf
    vol = excess.std(ddof=1)
    if vol == 0 or pd.isna(vol):
        return float("nan")
    return float(excess.mean() / vol * np.sqrt(252))
