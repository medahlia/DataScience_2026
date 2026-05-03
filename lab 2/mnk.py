import mean_squared_error
import requests
import pandas as pd
import numpy as np
import math as mt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, random, warnings
warnings.filterwarnings("ignore")


def mnk_poly(S0: np.ndarray, degree: int = 2) -> tuple:
    """
    МНК поліноміальна регресія довільного ступеня.
    Повертає: (Yout, coefficients)
    """
    n = len(S0)
    F = np.zeros((n, degree + 1))
    for d in range(degree + 1):
        F[:, d] = np.arange(n) ** d
    Yin = S0.reshape(-1, 1)
    FT = F.T
    C = np.linalg.inv(FT @ F) @ FT @ Yin
    Yout = F @ C
    return Yout, C


def mnk_extrapol_poly(S0: np.ndarray, koef: int, degree: int = 2) -> np.ndarray:
    """МНК прогноз на koef точок вперед."""
    n = len(S0)
    F = np.zeros((n, degree + 1))
    for d in range(degree + 1):
        F[:, d] = np.arange(n) ** d
    Yin = S0.reshape(-1, 1)
    FT = F.T
    C = np.linalg.inv(FT @ F) @ FT @ Yin
    n_full = n + koef
    F_full = np.zeros((n_full, degree + 1))
    for d in range(degree + 1):
        F_full[:, d] = np.arange(n_full) ** d
    Yout = F_full @ C
    return Yout


def r2_score_mnk(prices: np.ndarray, Yout: np.ndarray, label: str = "") -> float:
    n = len(prices)
    residuals = prices - Yout[:n, 0]
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((prices - np.mean(prices)) ** 2)
    R2 = 1 - ss_res / ss_tot
    if label:
        print(f"[{label}] R² = {R2:.6f}")
    return R2


def optimize_polynomial_degree(prices: np.ndarray,
                               max_degree: int = 6,
                               test_ratio: float = 0.2) -> int:
    """
    Підбір оптимального ступеня полінома за показниками:
      - MSE (mean squared error) на тестовій вибірці
      - MAE (mean absolute error)
      - R² на тестовій вибірці
      - AIC (Akaike Information Criterion) — штраф за складність

    Критерій вибору: мінімальний AIC (баланс якості та складності).
    """
    print("=" * 60)
    print("5. Оптимізація моделі — вибір ступеня полінома")
    n = len(prices)
    n_train = int(n * (1 - test_ratio))
    train = prices[:n_train]
    test = prices[n_train:]

    results = []
    print(f"{'Ступінь':>8} {'MSE_test':>12} {'MAE_test':>12} {'R2_train':>10} {'AIC':>12}")
    print("-" * 60)

    for degree in range(1, max_degree + 1):
        # навчання на train
        F_train = np.array([[i ** d for d in range(degree + 1)] for i in range(n_train)])
        C = np.linalg.lstsq(F_train, train, rcond=None)[0]

        # оцінка на test
        F_test = np.array([[(n_train + i) ** d for d in range(degree + 1)] for i in range(len(test))])
        pred_test = F_test @ C

        mse = mean_squared_error(test, pred_test)
        mae = mean_absolute_error(test, pred_test)

        # R² на train
        pred_train = F_train @ C
        r2_train = r2_score_mnk(train, pred_train.reshape(-1, 1))

        # AIC: n*ln(MSE) + 2*(degree+1)
        aic = n_train * np.log(np.mean((train - pred_train) ** 2) + 1e-10) + 2 * (degree + 1)

        results.append({"degree": degree, "mse": mse, "mae": mae, "r2": r2_train, "aic": aic})
        print(f"{degree:>8} {mse:>12.4f} {mae:>12.4f} {r2_train:>10.6f} {aic:>12.4f}")

    best = min(results, key=lambda x: x["aic"])
    print(f"\n✔ Оптимальний ступінь полінома: {best['degree']} (AIC={best['aic']:.4f})")

    # графік MSE та AIC за ступенями
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    degrees = [r["degree"] for r in results]
    axes[0].plot(degrees, [r["mse"] for r in results], "o-", color="steelblue")
    axes[0].axvline(best["degree"], color="red", linestyle="--", label=f"Оптимум: {best['degree']}")
    axes[0].set_title("MSE на тестовій вибірці")
    axes[0].set_xlabel("Ступінь полінома")
    axes[0].set_ylabel("MSE")
    axes[0].legend()

    axes[1].plot(degrees, [r["aic"] for r in results], "o-", color="crimson")
    axes[1].axvline(best["degree"], color="steelblue", linestyle="--", label=f"Оптимум: {best['degree']}")
    axes[1].set_title("AIC (штраф за складність)")
    axes[1].set_xlabel("Ступінь полінома")
    axes[1].set_ylabel("AIC")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("model_optimization.png", dpi=120)
    plt.close()
    print(f"Графік збережено: {os.path.abspath('model_optimization.png')}")

    return best["degree"]