"""
quality.py — Показники якості МНК-моделі та автоматичний вибір ступеня полінома.

Метрики:
  • MSE  — Mean Squared Error
  • MAE  — Mean Absolute Error
  • R²   — Coefficient of Determination
  • AIC  — Akaike Information Criterion (штраф за складність)
  • BIC  — Bayesian Information Criterion

Вибір ступеня: перебір degree=1..MAX_DEGREE, вибір за мінімальним AIC.
"""
from __future__ import annotations
import numpy as np
from mnk import mnk_poly

MAX_DEGREE = 6


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def aic(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """AIC = n·ln(MSE) + 2·k   (k — кількість параметрів = degree+1)."""
    n = len(y_true)
    mse_val = mse(y_true, y_pred)
    if mse_val <= 0:
        return -np.inf
    return n * np.log(mse_val) + 2 * k


def bic(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """BIC = n·ln(MSE) + k·ln(n)."""
    n = len(y_true)
    mse_val = mse(y_true, y_pred)
    if mse_val <= 0:
        return -np.inf
    return n * np.log(mse_val) + k * np.log(n)


def evaluate_degree(prices: np.ndarray, degree: int) -> dict:
    """Обчислює всі метрики для МНК-полінома заданого ступеня."""
    Yout, _ = mnk_poly(prices, degree)
    y_pred = Yout[:, 0]
    k = degree + 1
    return {
        "degree": degree,
        "MSE":    mse(prices, y_pred),
        "MAE":    mae(prices, y_pred),
        "R2":     r2(prices, y_pred),
        "AIC":    aic(prices, y_pred, k),
        "BIC":    bic(prices, y_pred, k),
    }


def select_best_degree(prices: np.ndarray, criterion: str = "AIC") -> int:
    """
    Перебирає степені 1..MAX_DEGREE, повертає найкращий за обраним критерієм.
    criterion: 'AIC' | 'BIC' | 'MSE' | 'MAE'  (мінімізація)
               'R2'  — максимізація
    """
    print("=" * 60)
    print(f"E. Оптимізація ступеня полінома (критерій: {criterion})")
    print(f"{'degree':>7} {'MSE':>12} {'MAE':>10} {'R2':>8} {'AIC':>12} {'BIC':>12}")
    print("-" * 65)

    results = []
    for deg in range(1, MAX_DEGREE + 1):
        m = evaluate_degree(prices, deg)
        results.append(m)
        print(f"{deg:>7} {m['MSE']:>12.4f} {m['MAE']:>10.4f} "
              f"{m['R2']:>8.6f} {m['AIC']:>12.2f} {m['BIC']:>12.2f}")

    if criterion == "R2":
        best = max(results, key=lambda x: x["R2"])
    else:
        best = min(results, key=lambda x: x[criterion])

    print(f"\n  ✔ Оптимальний ступінь полінома: {best['degree']}  "
          f"({criterion}={best[criterion]:.4f})")
    return best["degree"]


def print_quality_report(prices: np.ndarray, degree: int, label: str = "") -> dict:
    """Виводить повний звіт якості для обраного ступеня."""
    m = evaluate_degree(prices, degree)
    print("=" * 60)
    print(f"F. Показники якості моделі{' (' + label + ')' if label else ''}")
    print(f"   Ступінь полінома: {degree}")
    print(f"   MSE  = {m['MSE']:.4f}")
    print(f"   MAE  = {m['MAE']:.4f}")
    print(f"   R²   = {m['R2']:.6f}")
    print(f"   AIC  = {m['AIC']:.2f}")
    print(f"   BIC  = {m['BIC']:.2f}")
    return m