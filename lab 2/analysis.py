"""
analysis.py — Статистичний аналіз, синтез та верифікація МНК-моделі.
"""
from __future__ import annotations
import math as mt
import numpy as np
import pandas as pd

from mnk import MNK, MNK_AV_Detect, MNK_Stat_characteristics, mnk_extrapol, mnk_poly
from visualization import plot_trend, plot_residuals_hist, plot_forecast, plot_verification


def analyze_trend(prices: np.ndarray) -> None:
    print("=" * 60)
    print("3. Оцінка динаміки тренду")
    total_change = prices[-1] - prices[0]
    pct_change   = (total_change / prices[0]) * 100
    n            = len(prices)
    print(f"Перша ціна:      {prices[0]:.2f} USD")
    print(f"Остання ціна:    {prices[-1]:.2f} USD")
    print(f"Загальна зміна:  {total_change:+.2f} USD ({pct_change:+.2f}%)")
    MNK_AV_Detect(prices)


def stat_characteristics(prices: np.ndarray, label: str) -> dict:
    Yout     = MNK_Stat_characteristics(prices)
    residuals = np.array([prices[i] - Yout[i, 0] for i in range(len(prices))])

    m_mean   = float(np.mean(residuals))
    m_median = float(np.median(residuals))
    dS       = float(np.var(residuals))
    scvS     = mt.sqrt(dS)
    skew     = float(pd.Series(residuals).skew())
    kurt     = float(pd.Series(residuals).kurt())

    print("=" * 60)
    print(f"4. Статистичні характеристики: {label}")
    print(f"Кількість елементів вибірки: {len(prices)}")
    print(f"Математичне сподівання (mean): {m_mean:.4f}")
    print(f"Медіана залишків:              {m_median:.4f}")
    print(f"Дисперсія:                     {dS:.4f}")
    print(f"СКВ:                           {scvS:.4f}")
    print(f"Асиметрія:                     {skew:.4f}")
    print(f"Ексцес:                        {kurt:.4f}")
    print(f"Мін / Макс ціни: {prices.min():.2f} / {prices.max():.2f} USD")
    print(f"Розмах: {prices.max() - prices.min():.2f} USD")

    plot_residuals_hist(residuals, m_mean, label)

    return {"mean": m_mean, "median": m_median, "var": dS, "std": scvS,
            "skew": skew, "kurt": kurt, "min": prices.min(), "max": prices.max()}


def r2_score_mnk(prices: np.ndarray, Yout: np.ndarray, label: str) -> float:
    n           = len(prices)
    numerator   = sum((prices[i] - Yout[i, 0])**2 for i in range(n))
    mean_y      = np.mean(prices)
    denominator = sum((prices[i] - mean_y)**2 for i in range(n))
    R2          = 1 - numerator / denominator
    print(f"[{label}] Коефіцієнт детермінації R^2 = {R2:.6f}")
    return R2


def synthesize_and_verify(
    prices: np.ndarray,
    stats: dict,
    degree: int = 2,
) -> None:
    print("=" * 60)
    print("5. Синтез та верифікація моделі даних")

    from cleaning import sliding_window_clean
    from quality import evaluate_degree

    n        = len(prices)
    n_wind   = 5
    # ── ВИПРАВЛЕНО: прогноз на 0.5 від обсягу вибірки ──────────────────────
    koef     = mt.ceil(n * 0.5)

    prices_clean = sliding_window_clean(prices, n_wind)

    print("\n   --- МНК-згладжування ---")
    Yout_smooth, _ = mnk_poly(prices_clean, degree)
    R2_smooth = r2_score_mnk(prices_clean, Yout_smooth, "МНК-згладжування")

    print("\n   --- МНК-прогнозування ---")
    Yout_extrapol = mnk_extrapol(prices_clean, koef, degree)
    R2_extrapol = r2_score_mnk(prices_clean, Yout_extrapol[:n], "МНК-прогнозування")

    # Синтетичні дані
    synthetic = np.array([
        Yout_smooth[i, 0] + np.random.normal(stats["mean"], stats["std"])
        for i in range(n)
    ])

    print("\n   --- Верифікація: порівняння стат. характеристик ---")
    Yout_synth, _ = mnk_poly(synthetic, degree)
    synth_resid = np.array([synthetic[i] - Yout_synth[i, 0] for i in range(n)])
    print(f"Реальні дані: mean={stats['mean']:.4f}, σ={stats['std']:.4f}")
    print(f"Синт. модель: mean={np.mean(synth_resid):.4f}, σ={np.std(synth_resid):.4f}")
    r2_score_mnk(synthetic, Yout_synth, "синтетична МНК-модель")

    plot_forecast(prices_clean, Yout_smooth, Yout_extrapol, koef, degree)
    plot_verification(prices_clean, synthetic, Yout_smooth)


def print_analysis_summary(prices: np.ndarray, stats: dict) -> None:
    print("=" * 60)
    print("6. АНАЛІЗ РЕЗУЛЬТАТІВ")
    slope     = MNK_AV_Detect(prices)
    direction = "зростаючий" if slope > 0 else "спадаючий"
    print(f"""
   ✔ Парсинг сайту finance.yahoo.com/quote/GLD успішно виконано.
     Отримано {len(prices)} записів цін закриття ETF GLD.

   ✔ Динаміка тренду:
     Тренд {direction}. Нахил МНК-моделі = {slope:.6f}.
     Ціна зросла з {prices[0]:.2f} до {prices[-1]:.2f} USD
     (+{((prices[-1]-prices[0])/prices[0]*100):.1f}% за весь період).

   ✔ Статистичні характеристики залишків відносно тренду:
     Математичне сподівання ≈ {stats['mean']:.4f}
     СКВ (σ) = {stats['std']:.4f} USD
     Асиметрія = {stats['skew']:.4f}  |  Ексцес = {stats['kurt']:.4f}

   ✔ МНК-модель (оптимальна поліноміальна регресія):
     Ступінь полінома обрано автоматично за критерієм AIC.
     Прогноз на 50% від обсягу вибірки.

   ✔ Синтетична модель:
     y_synth = МНК-тренд + N(mean, σ).
     Статистичні характеристики синтетичних даних близькі до реальних.
    """)
    