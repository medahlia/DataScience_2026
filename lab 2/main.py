"""
main.py — Головний файл лабораторної роботи з Data Science.
Аналіз цін ETF GLD (Gold ETF).

Структура проєкту:
  parser.py           — парсинг Yahoo Finance та збереження CSV
  mnk.py              — МНК (LSM) поліноміальна регресія
  anomaly_injection.py — додавання аномальних вимірів
  anomaly_detection.py — виявлення аномалій (5 алгоритмів + grid search)
  cleaning.py         — очищення від аномалій
  quality.py          — MSE / MAE / R² / AIC / BIC, вибір ступеня полінома
  filters.py          — α-β та α-β-γ рекурентні фільтри
  visualization.py    — всі графіки
  analysis.py         — статистика, синтез, верифікація, звіт
"""
import math as mt
import numpy as np

# ── Модулі проєкту ────────────────────────────────────────────────────────────
from parser   import parse_gold_prices, clean_dataframe, save_to_csv
from mnk      import MNK_Stat_characteristics, MNK_AV_Detect

from analysis     import analyze_trend, stat_characteristics, synthesize_and_verify, print_analysis_summary
from visualization import plot_trend

from anomalies import inject_anomalies, save_anomaly_csv
from anomaly_detection import grid_search_params, compare_detectors
from cleaning          import clean_prices
from quality           import select_best_degree, print_quality_report, evaluate_degree
from filters           import auto_filter, tune_filter
from visualization     import (
    plot_anomalies, plot_cleaning,
    plot_degree_selection, plot_filter,
)


# ============================================================
# ГОЛОВНИЙ БЛОК
# ============================================================

if __name__ == "__main__":

    # ── 1. Парсинг ────────────────────────────────────────────────────────────
    df_raw   = parse_gold_prices()
    df_clean = clean_dataframe(df_raw)

    # ── 2. Збереження у CSV ───────────────────────────────────────────────────
    save_to_csv(df_clean, "gold_prices.csv")

    prices = df_clean["Close"].values.astype(float)

    # ── 3. Оцінка динаміки тренду ─────────────────────────────────────────────
    analyze_trend(prices)
    Yout_trend = MNK_Stat_characteristics(prices)
    plot_trend(prices, Yout_trend,
               "Динаміка цін ETF GLD та МНК-тренд", "trend_analysis.png")

    # ── 4. Статистичні характеристики ─────────────────────────────────────────
    stats = stat_characteristics(prices, "ETF GLD (ціна закриття)")

    # ── 5. Синтез та верифікація МНК-моделі ───────────────────────────────────
    synthesize_and_verify(prices, stats)

    # ── 6. Аналіз результатів ─────────────────────────────────────────────────
    print_analysis_summary(prices, stats)

    # =========================================================
    # НОВІ БЛОКИ ЛАБОРАТОРНОЇ РОБОТИ
    # =========================================================

    # ── A. Внесення аномалій ──────────────────────────────────────────────────
    print("=" * 60)
    print("A. Внесення аномальних вимірів")
    df_anomaly, anomaly_idx = inject_anomalies(df_clean, seed=42)
    save_anomaly_csv(df_anomaly, "gold_prices_anomaly.csv")
    prices_anomaly = df_anomaly["Close"].values.astype(float)

    # ── B. Навчання параметрів детекторів (grid search) ───────────────────────
    true_anomaly_set = set(anomaly_idx)
    grid_results = grid_search_params(prices_anomaly, true_anomaly_set)

    # ── C. Порівняння та вибір найкращого детектора ───────────────────────────
    best_detector, detected_anomalies = compare_detectors(
        prices_anomaly, true_anomaly_set, grid_results
    )
    plot_anomalies(prices, prices_anomaly, anomaly_idx,
                   detected_anomalies, "anomaly_detection.png")

    # ── D. Очищення від аномалій ──────────────────────────────────────────────
    prices_cleaned = clean_prices(prices_anomaly, detected_anomalies,
                                  strategy="interpolate")
    plot_cleaning(prices_anomaly, prices_cleaned, anomaly_idx,
                  "cleaning_result.png")

    # ── E. Оптимізація ступеня полінома (MSE, MAE, R², AIC, BIC) ─────────────
    from quality import MAX_DEGREE, evaluate_degree as _eval_deg
    metrics_list = [_eval_deg(prices_cleaned, d) for d in range(1, MAX_DEGREE + 1)]
    best_degree  = select_best_degree(prices_cleaned, criterion="AIC")
    quality_info = print_quality_report(prices_cleaned, best_degree, "очищені дані")
    plot_degree_selection(metrics_list, best_degree, "degree_selection.png")

    # ── G. Рекурентне згладжування: α-β або α-β-γ ────────────────────────────
    r2_val   = quality_info["R2"]
    filter_type = "abg" if r2_val >= 0.90 else "ab"
    best_filter_params = tune_filter(prices_cleaned, filter_type=filter_type)
    filtered_prices, filter_name = auto_filter(prices_cleaned, r2_val)
    plot_filter(prices_cleaned, filtered_prices, filter_name, "filter_result.png")

    print("=" * 60)
    print("Лабораторну роботу завершено.")
    print("=" * 60)