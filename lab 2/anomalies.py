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
# ══  П.1 — ГЕНЕРАЦІЯ АНОМАЛЬНИХ ВИМІРІВ  ══
# ─────────────────────────────────────────────────────────────

def inject_anomalies(prices: np.ndarray, n_anomalies: int = 40, amplitude_k: float = 4.0, seed: int = 42) -> tuple:
    """
    Вносить n_anomalies аномальних вимірів у вибірку.

    Три типи аномалій:
      - spike_up   : різкий стрибок вгору  (amplitude_k * std)
      - spike_down : різкий стрибок вниз
      - drift      : повільний дрейф (5 поспіль точок зміщені)

    Повертає: (prices_with_anomalies, anomaly_indices, original_values)
    """
    rng = np.random.default_rng(seed)
    sigma = float(np.std(prices))
    dirty = prices.copy()
    n = len(prices)

    indices = []
    originals = []

    # spike аномалії
    spike_idx = rng.choice(n, size=n_anomalies, replace=False)
    for idx in spike_idx:
        orig = dirty[idx]
        direction = rng.choice([-1, 1])
        dirty[idx] = orig + direction * amplitude_k * sigma * rng.uniform(0.8, 1.2)
        indices.append(int(idx))
        originals.append(float(orig))

    # drift-аномалія (один блок з 5 точок)
    drift_start = int(rng.integers(100, n - 10))
    drift_amp = amplitude_k * sigma * 0.6
    for k in range(5):
        idx = drift_start + k
        if idx not in indices:
            orig = dirty[idx]
            dirty[idx] = orig + drift_amp * (k + 1) / 5
            indices.append(idx)
            originals.append(float(orig))

    print("=" * 60)
    print(f"П.1 — Внесено аномалій: {len(indices)}")
    print(f"      Типи: spike_up / spike_down (n={n_anomalies}), drift (5 точок)")
    print(f"      Амплітуда: ±{amplitude_k:.1f}σ  (σ={sigma:.2f} USD)")
    return dirty, sorted(indices), originals


def save_anomaly_csv(df_clean: pd.DataFrame, prices_anom: np.ndarray,
                     anomaly_indices: list, anomaly_types: list,
                     filename: str = "gold_prices_with_anomalies.csv") -> None:
    df_anom = df_clean.copy()
    df_anom["Close"] = prices_anom
    df_anom["is_anomaly"] = False
    df_anom["anomaly_type"] = ""
    for idx, atype in zip(anomaly_indices, anomaly_types):
        df_anom.loc[idx, "is_anomaly"] = True
        df_anom.loc[idx, "anomaly_type"] = atype
    df_anom.to_csv(filename, index=False, encoding="utf-8-sig")
    print("=" * 60)
    print("3. Збереження даних з аномаліями")
    print(f"Файл збережено: {os.path.abspath(filename)}")
    print(f"Введено аномалій: {len(anomaly_indices)} ({len(anomaly_indices)/len(prices_anom)*100:.1f}%)")
    print(f"Типи: { {t: anomaly_types.count(t) for t in set(anomaly_types)} }")


def plot_anomalies(prices_orig: np.ndarray, prices_anom: np.ndarray,
                   anomaly_indices: list, filename: str = "anomalies_injected.png") -> None:
    plt.figure(figsize=(13, 5))
    plt.plot(prices_orig, color="steelblue", linewidth=1, label="Оригінальні дані")
    plt.plot(prices_anom, color="gray", linewidth=0.8, alpha=0.7, label="З аномаліями")
    plt.scatter(anomaly_indices, prices_anom[anomaly_indices],
                color="red", zorder=5, s=40, label="Аномалії (введені)")
    plt.title("Введення аномальних вимірів до часового ряду ETF GLD", fontsize=13)
    plt.xlabel("Індекс спостереження")
    plt.ylabel("Ціна закриття, USD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Графік збережено: {os.path.abspath(filename)}")


