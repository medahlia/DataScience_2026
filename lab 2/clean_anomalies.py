import requests
import pandas as pd
import numpy as np
import math as mt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, random, warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# ══  П.2 — ВИЯВЛЕННЯ ТА ОЧИЩЕННЯ АНОМАЛІЙ (Median / MAD)  ══
# ─────────────────────────────────────────────────────────────

def detect_clean_median(prices: np.ndarray, window: int = 11, threshold_k: float = 3.0) -> tuple:
    """
    Виявляє та усуває аномалії методом ковзного медіанного фільтра (MAD).

    Алгоритм:
      1. Для кожної точки обчислюємо локальну медіану у вікні ±window//2.
      2. Відхилення від медіани: dev = |x - median|.
      3. MAD (Median Absolute Deviation) = median(|dev|).
      4. Точка — аномалія, якщо dev > threshold_k * MAD / 0.6745.
         (0.6745 — нормуючий коефіцієнт для нормального розподілу)
      5. Замінюємо аномалії локальною медіаною.

    Повертає: (cleaned_prices, detected_mask)
    """
    n = len(prices)
    cleaned = prices.copy()
    mask = np.zeros(n, dtype=bool)
    half = window // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        win = prices[lo:hi]
        med = np.median(win)
        mad = np.median(np.abs(win - med))
        thr = threshold_k * mad / 0.6745
        if abs(prices[i] - med) > thr and thr > 1e-9:
            mask[i] = True
            cleaned[i] = med  # заміна медіаною вікна

    detected = int(np.sum(mask))
    print("=" * 60)
    print(f"П.2 — Median/MAD фільтр (вікно={window}, k={threshold_k})")
    print(f"      Виявлено аномалій : {detected}")
    print(f"      Правильно виявлено: перевірте графік")
    return cleaned, mask


def clean_anomalies_interpolation(prices: np.ndarray,
                                   anomaly_mask: np.ndarray) -> np.ndarray:
    """
    Очищення аномалій: лінійна інтерполяція за сусідніми нормальними точками.
    """
    prices_cleaned = prices.copy().astype(float)
    prices_cleaned[anomaly_mask] = np.nan
    # pandas для зручної інтерполяції
    s = pd.Series(prices_cleaned)
    s.interpolate(method="linear", inplace=True)
    s.bfill(inplace=True)
    s.ffill(inplace=True)
    return s.values


def anomaly_cleaning_report(prices_anom: np.ndarray, prices_cleaned: np.ndarray,
                            anomaly_mask: np.ndarray, true_indices: list) -> None:
    detected = np.where(anomaly_mask)[0]
    true_set = set(true_indices)
    detected_set = set(detected)

    tp = len(true_set & detected_set)
    fp = len(detected_set - true_set)
    fn = len(true_set - detected_set)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    print("=" * 60)
    print("4. Виявлення та очищення аномалій (IQR + Z-score)")
    print(f"Виявлено аномалій: {len(detected)}")
    print(f"Істинно позитивних (TP): {tp}")
    print(f"Хибно позитивних  (FP): {fp}")
    print(f"Хибно негативних  (FN): {fn}")
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")


def plot_cleaning(prices_orig, prices_anom, prices_cleaned, anomaly_mask,
                  filename="anomaly_cleaning.png"):
    detected_idx = np.where(anomaly_mask)[0]
    plt.figure(figsize=(13, 5))
    plt.plot(prices_orig, color="steelblue", linewidth=1, label="Оригінальні дані", alpha=0.7)
    plt.plot(prices_anom, color="gray", linewidth=0.8, alpha=0.5, label="З аномаліями")
    plt.plot(prices_cleaned, color="green", linewidth=1.5, label="Очищені дані")
    plt.scatter(detected_idx, prices_anom[detected_idx],
                color="red", zorder=5, s=40, label="Виявлені аномалії")
    plt.title("Очищення аномалій: IQR + Z-score", fontsize=13)
    plt.xlabel("Індекс спостереження")
    plt.ylabel("Ціна, USD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Графік збережено: {os.path.abspath(filename)}")
