"""
anomaly_detection.py — Складні алгоритми виявлення аномалій з «навчанням» параметрів.

Реалізовані методи:
  1. IQR з адаптивним вікном (sliding IQR)
  2. Z-score з ковзним середнім та СКВ (robust rolling Z-score)
  3. GESD (Generalized ESD Test) — статистичний тест на outliers
  4. LOF-inspired score (спрощений Local Outlier Factor на 1D ряді)
  5. MAD-score (Median Absolute Deviation) — стійкий до outliers
  6. Ансамблевий метод: об'єднання голосів усіх алгоритмів

«Навчання параметрів»:
  - grid_search_params() підбирає поріг / вікно для кожного алгоритму,
    мінімізуючи F1-score відносно відомих аномальних індексів (gold_idx).
  - compare_detectors() порівнює всі методи і рекомендує найкращий.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from itertools import product as iproduct
from typing import Callable


# ── Утиліти ──────────────────────────────────────────────────────────────────

def _f1(detected: set[int], true_pos: set[int], n: int) -> float:
    """F1-score для детектора аномалій."""
    tp = len(detected & true_pos)
    fp = len(detected - true_pos)
    fn = len(true_pos - detected)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _precision_recall_f1(detected: set[int], true_pos: set[int]) -> tuple[float, float, float]:
    tp = len(detected & true_pos)
    fp = len(detected - true_pos)
    fn = len(true_pos - detected)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ── 1. Sliding IQR ───────────────────────────────────────────────────────────

def sliding_iqr(prices: np.ndarray, window: int = 60, multiplier: float = 2.5) -> set[int]:
    """
    Sliding-window IQR: на кожному вікні обчислюємо Q1/Q3,
    точка — аномалія якщо виходить за [Q1 - k·IQR, Q3 + k·IQR].
    """
    n = len(prices)
    anomalies: set[int] = set()
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half)
        window_data = prices[lo:hi]
        q1, q3 = np.percentile(window_data, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        if prices[i] < q1 - multiplier * iqr or prices[i] > q3 + multiplier * iqr:
            anomalies.add(i)
    return anomalies


# ── 2. Robust Rolling Z-score ────────────────────────────────────────────────

def rolling_zscore(prices: np.ndarray, window: int = 30, threshold: float = 3.0) -> set[int]:
    """
    Z-score на ковзному вікні: z = (x - mean) / std.
    Використовується trimmed mean для стійкості.
    """
    n = len(prices)
    anomalies: set[int] = set()
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        w = prices[lo:hi]
        # Trimmed stats (відкидаємо 10% крайніх значень)
        pct_lo, pct_hi = np.percentile(w, [10, 90])
        trimmed = w[(w >= pct_lo) & (w <= pct_hi)]
        if len(trimmed) < 3:
            trimmed = w
        mu = np.mean(trimmed)
        sigma = np.std(trimmed)
        if sigma == 0:
            continue
        z = abs(prices[i] - mu) / sigma
        if z > threshold:
            anomalies.add(i)
    return anomalies


# ── 3. GESD (Generalized ESD Test) ───────────────────────────────────────────

def gesd_test(prices: np.ndarray, max_outliers_frac: float = 0.05,
              alpha: float = 0.05) -> set[int]:
    """
    Generalized Extreme Studentized Deviate test.
    Ітеративно видаляє найвіддаленіші від середнього значення.
    """
    from scipy import stats as sp_stats

    data = prices.copy()
    n = len(data)
    max_outliers = max(1, int(n * max_outliers_frac))
    indices = list(range(n))
    removed: list[tuple[int, float]] = []

    for _ in range(max_outliers):
        mu = np.mean(data)
        sigma = np.std(data, ddof=1)
        if sigma == 0:
            break
        deviations = np.abs(data - mu)
        worst_local = int(np.argmax(deviations))
        R = deviations[worst_local] / sigma
        removed.append((indices[worst_local], R))
        data = np.delete(data, worst_local)
        indices.pop(worst_local)

    # Критичне значення: t-розподіл
    anomalies: set[int] = set()
    n_cur = n
    for k, (idx, R) in enumerate(removed):
        p = alpha / (2 * (n_cur - k))
        t_crit = sp_stats.t.ppf(1 - p, df=n_cur - k - 2)
        lam = ((n_cur - k - 1) * t_crit
               / math.sqrt((n_cur - k - 2 + t_crit**2) * (n_cur - k)))
        if R > lam:
            anomalies.add(idx)
    return anomalies


# ── 4. LOF-inspired 1D score ─────────────────────────────────────────────────

def lof_1d(prices: np.ndarray, k: int = 20, threshold: float = 1.8) -> set[int]:
    """
    Спрощений LOF для 1D: оцінюємо «щільність» кожної точки
    через середню відстань до k-найближчих сусідів.
    LOF = відношення середньої локальної щільності сусідів до власної щільності.
    """
    n = len(prices)
    k = min(k, n - 1)
    anomalies: set[int] = set()

    # Reach-distance та lrd
    reach_dist = np.zeros(n)
    lrd = np.zeros(n)

    for i in range(n):
        dists = np.abs(prices - prices[i])
        dists[i] = np.inf
        neighbors = np.argsort(dists)[:k]
        k_dist = dists[neighbors[-1]]
        # reach-distance = max(k-dist(neighbor), actual_dist)
        reach_dist[i] = np.mean(np.maximum(
            [np.sort(np.abs(prices - prices[j]))[:k][-1] for j in neighbors],
            dists[neighbors]
        ))
        lrd[i] = 1.0 / (reach_dist[i] + 1e-12)

    for i in range(n):
        dists = np.abs(prices - prices[i])
        dists[i] = np.inf
        neighbors = np.argsort(dists)[:k]
        lof = np.mean(lrd[neighbors]) / (lrd[i] + 1e-12)
        if lof > threshold:
            anomalies.add(i)

    return anomalies


# ── 5. MAD-score ─────────────────────────────────────────────────────────────

def mad_score(prices: np.ndarray, threshold: float = 3.5) -> set[int]:
    """
    Median Absolute Deviation — стійкий до outliers аналог Z-score.
    Modified Z-score: M_i = 0.6745 * (x_i - median) / MAD
    """
    median = np.median(prices)
    mad = np.median(np.abs(prices - median))
    if mad == 0:
        mad = np.mean(np.abs(prices - median))
    modified_z = 0.6745 * np.abs(prices - median) / (mad + 1e-12)
    return set(int(i) for i in np.where(modified_z > threshold)[0])


# ── 6. Ensemble detector ─────────────────────────────────────────────────────

def ensemble_detect(
    prices: np.ndarray,
    min_votes: int = 3,
    iqr_window: int = 60, iqr_mult: float = 2.5,
    rz_window: int = 30, rz_thresh: float = 3.0,
    gesd_frac: float = 0.05,
    lof_k: int = 20, lof_thresh: float = 1.8,
    mad_thresh: float = 3.5,
) -> set[int]:
    """
    Ансамбль: точка вважається аномальною якщо ≥ min_votes алгоритмів її позначили.
    """
    votes: dict[int, int] = {}

    detectors = [
        sliding_iqr(prices, iqr_window, iqr_mult),
        rolling_zscore(prices, rz_window, rz_thresh),
        gesd_test(prices, gesd_frac),
        lof_1d(prices, lof_k, lof_thresh),
        mad_score(prices, mad_thresh),
    ]

    for det in detectors:
        for idx in det:
            votes[idx] = votes.get(idx, 0) + 1

    return {idx for idx, v in votes.items() if v >= min_votes}


# ── Grid-search для навчання параметрів ──────────────────────────────────────

def grid_search_params(
    prices: np.ndarray,
    true_anomalies: set[int],
    verbose: bool = True,
) -> dict:
    """
    Підбір оптимальних параметрів для кожного детектора методом перебору сітки.
    Критерій: максимальний F1-score.

    Повертає словник {detector_name: {best_params, best_f1}}.
    """
    n = len(prices)
    results: dict = {}

    print("=" * 60)
    print("B. Навчання параметрів детекторів аномалій (Grid Search)")

    # -- Sliding IQR ----------------------------------------------------------
    best_f1, best_p = 0.0, {}
    for window, mult in iproduct([30, 60, 90, 120], [1.5, 2.0, 2.5, 3.0, 3.5]):
        det = sliding_iqr(prices, window, mult)
        f1 = _f1(det, true_anomalies, n)
        if f1 > best_f1:
            best_f1, best_p = f1, {"window": window, "multiplier": mult}
    results["sliding_iqr"] = {"params": best_p, "f1": best_f1}
    if verbose:
        print(f"  sliding_iqr   → best F1={best_f1:.4f}  params={best_p}")

    # -- Rolling Z-score ------------------------------------------------------
    best_f1, best_p = 0.0, {}
    for window, thresh in iproduct([20, 30, 50, 70], [2.0, 2.5, 3.0, 3.5, 4.0]):
        det = rolling_zscore(prices, window, thresh)
        f1 = _f1(det, true_anomalies, n)
        if f1 > best_f1:
            best_f1, best_p = f1, {"window": window, "threshold": thresh}
    results["rolling_zscore"] = {"params": best_p, "f1": best_f1}
    if verbose:
        print(f"  rolling_zscore → best F1={best_f1:.4f}  params={best_p}")

    # -- GESD -----------------------------------------------------------------
    best_f1, best_p = 0.0, {}
    for frac, alpha in iproduct([0.03, 0.05, 0.08, 0.10], [0.01, 0.05, 0.10]):
        det = gesd_test(prices, frac, alpha)
        f1 = _f1(det, true_anomalies, n)
        if f1 > best_f1:
            best_f1, best_p = f1, {"max_outliers_frac": frac, "alpha": alpha}
    results["gesd"] = {"params": best_p, "f1": best_f1}
    if verbose:
        print(f"  gesd           → best F1={best_f1:.4f}  params={best_p}")

    # -- LOF 1D ---------------------------------------------------------------
    best_f1, best_p = 0.0, {}
    for k, thresh in iproduct([10, 20, 30, 50], [1.5, 1.8, 2.0, 2.5, 3.0]):
        det = lof_1d(prices, k, thresh)
        f1 = _f1(det, true_anomalies, n)
        if f1 > best_f1:
            best_f1, best_p = f1, {"k": k, "threshold": thresh}
    results["lof_1d"] = {"params": best_p, "f1": best_f1}
    if verbose:
        print(f"  lof_1d         → best F1={best_f1:.4f}  params={best_p}")

    # -- MAD ------------------------------------------------------------------
    best_f1, best_p = 0.0, {}
    for thresh in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        det = mad_score(prices, thresh)
        f1 = _f1(det, true_anomalies, n)
        if f1 > best_f1:
            best_f1, best_p = f1, {"threshold": thresh}
    results["mad"] = {"params": best_p, "f1": best_f1}
    if verbose:
        print(f"  mad            → best F1={best_f1:.4f}  params={best_p}")

    # -- Ensemble (min_votes grid) -------------------------------------------
    best_f1, best_p = 0.0, {}
    for votes in [2, 3, 4]:
        det = ensemble_detect(prices, min_votes=votes)
        f1 = _f1(det, true_anomalies, n)
        if f1 > best_f1:
            best_f1, best_p = f1, {"min_votes": votes}
    results["ensemble"] = {"params": best_p, "f1": best_f1}
    if verbose:
        print(f"  ensemble       → best F1={best_f1:.4f}  params={best_p}")

    return results


def compare_detectors(
    prices: np.ndarray,
    true_anomalies: set[int],
    grid_results: dict,
) -> tuple[str, set[int]]:
    """
    Порівнює всі детектори за F1 і повертає назву переможця та його набір аномалій.
    """
    print("=" * 60)
    print("C. Порівняння детекторів аномалій")
    print(f"{'Метод':<18} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 48)

    best_name, best_f1, best_set = "", 0.0, set()

    for name, info in grid_results.items():
        params = info["params"]
        # Перезапустити з найкращими параметрами
        if name == "sliding_iqr":
            det = sliding_iqr(prices, **params)
        elif name == "rolling_zscore":
            det = rolling_zscore(prices, **params)
        elif name == "gesd":
            det = gesd_test(prices, **params)
        elif name == "lof_1d":
            det = lof_1d(prices, **params)
        elif name == "mad":
            det = mad_score(prices, **params)
        elif name == "ensemble":
            det = ensemble_detect(prices, **params)
        else:
            det = set()

        prec, rec, f1 = _precision_recall_f1(det, true_anomalies)
        print(f"{name:<18} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f}")

        if f1 > best_f1:
            best_f1, best_name, best_set = f1, name, det

    print(f"\n  ✔ Рекомендований детектор: {best_name}  (F1={best_f1:.4f})")
    return best_name, best_set