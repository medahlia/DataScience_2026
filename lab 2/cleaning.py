"""
cleaning.py — Очищення вхідних даних від аномальних вимірів.

Стратегії заміни:
  • interpolate  — лінійна інтерполяція між сусідами (за замовчуванням)
  • median_fill  — медіана локального вікна
  • mnk_fill     — значення МНК-тренду в даній точці
  • drop         — просто видалити рядки (змінює довжину!)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from mnk import mnk_poly


def clean_by_interpolation(prices: np.ndarray, anomaly_idx: set[int]) -> np.ndarray:
    """Лінійна інтерполяція замість аномальних точок."""
    s = pd.Series(prices.copy())
    s[list(anomaly_idx)] = np.nan
    s = s.interpolate(method="linear", limit_direction="both")
    return s.values


def clean_by_median(prices: np.ndarray, anomaly_idx: set[int], window: int = 10) -> np.ndarray:
    """Замінює аномалії медіаною сусіднього вікна."""
    result = prices.copy()
    n = len(prices)
    for idx in anomaly_idx:
        lo = max(0, idx - window // 2)
        hi = min(n, idx + window // 2 + 1)
        neighbors = [prices[j] for j in range(lo, hi) if j not in anomaly_idx]
        if neighbors:
            result[idx] = float(np.median(neighbors))
    return result


def clean_by_mnk(prices: np.ndarray, anomaly_idx: set[int], degree: int = 2) -> np.ndarray:
    """Замінює аномалії значенням МНК-тренду."""
    Yout, _ = mnk_poly(prices, degree)
    result = prices.copy()
    for idx in anomaly_idx:
        result[idx] = Yout[idx, 0]
    return result


def clean_prices(
    prices: np.ndarray,
    anomaly_idx: set[int],
    strategy: str = "interpolate",
    **kwargs,
) -> np.ndarray:
    """
    Уніфікований інтерфейс очищення.

    strategy: 'interpolate' | 'median_fill' | 'mnk_fill'
    """
    print("=" * 60)
    print(f"D. Очищення даних  (стратегія: {strategy})")
    print(f"   Аномальних точок для заміни: {len(anomaly_idx)}")

    if strategy == "interpolate":
        cleaned = clean_by_interpolation(prices, anomaly_idx)
    elif strategy == "median_fill":
        cleaned = clean_by_median(prices, anomaly_idx, **kwargs)
    elif strategy == "mnk_fill":
        cleaned = clean_by_mnk(prices, anomaly_idx, **kwargs)
    else:
        raise ValueError(f"Невідома стратегія: {strategy}")

    n_nan = int(np.sum(np.isnan(cleaned)))
    print(f"   NaN після очищення: {n_nan}")
    return cleaned


def sliding_window_clean(S0: np.ndarray, n_wind: int) -> np.ndarray:
    """
    Медіанне ковзне згладжування (оригінальна функція з main.py).
    Залишена тут для сумісності.
    """
    import math as mt
    n = len(S0)
    j_wind = mt.ceil(n - n_wind) + 1
    Midi = np.zeros(n)
    wind = np.zeros(n_wind)
    for j in range(j_wind):
        for i in range(n_wind):
            wind[i] = S0[j + i]
        Midi[j + n_wind - 1] = np.median(wind)
    S_clean = np.copy(S0)
    for j in range(n_wind, n):
        S_clean[j] = Midi[j]
    return S_clean