"""
visualization.py — Всі функції побудови графіків.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Палітра ───────────────────────────────────────────────────────────────────
C_REAL    = "#2C6FAC"
C_TREND   = "#C0392B"
C_PRED    = "#E67E22"
C_SYNTH   = "#27AE60"
C_ANOMALY = "#8E44AD"
C_CLEAN   = "#16A085"
C_FILTER  = "#F39C12"


def _save(fig, filename: str) -> None:
    fig.savefig(filename, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"   Графік збережено: {os.path.abspath(filename)}")


# ── Тренд ─────────────────────────────────────────────────────────────────────

def plot_trend(prices: np.ndarray, Yout: np.ndarray,
               title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(prices,        color=C_REAL,  lw=1,   label="Реальні ціни GLD")
    ax.plot(Yout[:, 0],    color=C_TREND, lw=2, ls="--",
            label="МНК-тренд (квадратична модель)")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Індекс спостереження")
    ax.set_ylabel("Ціна закриття, USD")
    ax.legend()
    fig.tight_layout()
    _save(fig, filename)


# ── Аномалії ─────────────────────────────────────────────────────────────────

def plot_anomalies(
    prices_clean: np.ndarray,
    prices_anomaly: np.ndarray,
    anomaly_idx: list[int],
    detected_idx: set[int],
    filename: str = "anomaly_detection.png",
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # Верхній: реальний vs аномальний
    axes[0].plot(prices_clean,   color=C_REAL,  lw=1, label="Оригінальні дані")
    axes[0].plot(prices_anomaly, color=C_ANOMALY, lw=1, alpha=0.7,
                 label="Дані з аномаліями")
    axes[0].scatter(anomaly_idx, prices_anomaly[anomaly_idx],
                    color="red", s=15, zorder=5, label="Введені аномалії")
    axes[0].set_title("Оригінальні vs аномальні дані", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].set_ylabel("Ціна, USD")

    # Нижній: виявлені аномалії
    detected = sorted(detected_idx)
    axes[1].plot(prices_anomaly, color=C_ANOMALY, lw=1, alpha=0.6,
                 label="Аномальні дані")
    if detected:
        axes[1].scatter(detected, prices_anomaly[detected],
                        color="orange", s=20, zorder=5,
                        label=f"Виявлені аномалії ({len(detected)})")
    axes[1].set_title("Виявлені аномалії (найкращий детектор)", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].set_xlabel("Індекс спостереження")
    axes[1].set_ylabel("Ціна, USD")

    fig.tight_layout()
    _save(fig, filename)


# ── Порівняння очищення ───────────────────────────────────────────────────────

def plot_cleaning(
    prices_anomaly: np.ndarray,
    prices_cleaned: np.ndarray,
    anomaly_idx: list[int],
    filename: str = "cleaning_result.png",
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(prices_anomaly, color=C_ANOMALY, lw=1, alpha=0.6,
            label="Дані з аномаліями")
    ax.plot(prices_cleaned, color=C_CLEAN, lw=1.5,
            label="Очищені дані (інтерполяція)")
    ax.scatter(anomaly_idx, prices_anomaly[anomaly_idx],
               color="red", s=12, zorder=5, label="Аномальні точки")
    ax.set_title("Результат очищення від аномалій", fontsize=13)
    ax.set_xlabel("Індекс спостереження")
    ax.set_ylabel("Ціна, USD")
    ax.legend()
    fig.tight_layout()
    _save(fig, filename)


# ── Вибір ступеня полінома ────────────────────────────────────────────────────

def plot_degree_selection(
    metrics: list[dict],
    best_degree: int,
    filename: str = "degree_selection.png",
) -> None:
    degrees = [m["degree"] for m in metrics]
    mse_vals = [m["MSE"] for m in metrics]
    mae_vals = [m["MAE"] for m in metrics]
    r2_vals  = [m["R2"] for m in metrics]
    aic_vals = [m["AIC"] for m in metrics]

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    def _subplot(ax, vals, ylabel, title, color):
        ax.plot(degrees, vals, "o-", color=color, lw=2)
        ax.axvline(best_degree, color="red", ls="--", lw=1.5, label=f"best={best_degree}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Ступінь полінома")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)

    _subplot(fig.add_subplot(gs[0, 0]), mse_vals, "MSE",  "MSE vs ступінь", "#2C6FAC")
    _subplot(fig.add_subplot(gs[0, 1]), mae_vals, "MAE",  "MAE vs ступінь", "#27AE60")
    _subplot(fig.add_subplot(gs[1, 0]), r2_vals,  "R²",   "R² vs ступінь",  "#C0392B")
    _subplot(fig.add_subplot(gs[1, 1]), aic_vals, "AIC",  "AIC vs ступінь", "#8E44AD")

    fig.suptitle("Оптимізація ступеня полінома МНК", fontsize=13, fontweight="bold")
    _save(fig, filename)


# ── МНК-прогноз ──────────────────────────────────────────────────────────────

def plot_forecast(
    prices_clean: np.ndarray,
    Yout_smooth: np.ndarray,
    Yout_extrapol: np.ndarray,
    koef: int,
    degree: int,
    filename: str = "mnk_trend_forecast.png",
) -> None:
    n = len(prices_clean)
    x_hist = np.arange(n)
    x_full = np.arange(n + koef)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x_hist, prices_clean, color=C_REAL, lw=1,
            label="Реальні дані (очищені)")
    ax.plot(x_hist, Yout_smooth[:, 0], color=C_TREND, lw=2, ls="--",
            label=f"МНК-тренд (degree={degree})")
    ax.plot(x_full[n:], Yout_extrapol[n:, 0], color=C_PRED, lw=2, ls=":",
            label=f"МНК-прогноз (+{koef} точок)")
    ax.axvline(n - 1, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_title("МНК: тренд та прогноз цін ETF GLD", fontsize=13)
    ax.set_xlabel("Індекс спостереження")
    ax.set_ylabel("Ціна закриття, USD")
    ax.legend()
    fig.tight_layout()
    _save(fig, filename)


# ── Верифікація: реальні vs синтетичні ───────────────────────────────────────

def plot_verification(
    prices_clean: np.ndarray,
    synthetic: np.ndarray,
    Yout_smooth: np.ndarray,
    filename: str = "verification_real_vs_synthetic.png",
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(prices_clean, color=C_REAL,  lw=1, alpha=0.8,  label="Реальні дані GLD")
    ax.plot(synthetic,    color=C_SYNTH, lw=1, alpha=0.6, ls="--",
            label="Синтетична модель")
    ax.plot(Yout_smooth[:, 0], color=C_TREND, lw=2, label="МНК-тренд")
    ax.set_title("Верифікація: реальні дані vs синтетична модель", fontsize=13)
    ax.set_xlabel("Індекс спостереження")
    ax.set_ylabel("Ціна, USD")
    ax.legend()
    fig.tight_layout()
    _save(fig, filename)


# ── Гістограма залишків ───────────────────────────────────────────────────────

def plot_residuals_hist(
    residuals: np.ndarray,
    mean_val: float,
    label: str,
    filename: str = "histogram_residuals.png",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=30, color=C_REAL, alpha=0.75, edgecolor="white")
    ax.axvline(mean_val, color=C_TREND, ls="--", label=f"mean={mean_val:.2f}")
    ax.set_title(f"Гістограма залишків: {label}", fontsize=12)
    ax.set_xlabel("Відхилення від МНК-тренду, USD")
    ax.set_ylabel("Частота")
    ax.legend()
    fig.tight_layout()
    _save(fig, filename)


# ── α-β / α-β-γ фільтр ───────────────────────────────────────────────────────

def plot_filter(
    prices: np.ndarray,
    filtered: np.ndarray,
    filter_name: str,
    filename: str = "filter_result.png",
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    axes[0].plot(prices,   color=C_REAL,   lw=1, alpha=0.8, label="Вхідні дані")
    axes[0].plot(filtered, color=C_FILTER, lw=2, label=f"{filter_name} фільтр")
    axes[0].set_title(f"Рекурентне згладжування: {filter_name} фільтр", fontsize=12)
    axes[0].legend()
    axes[0].set_ylabel("Ціна, USD")

    residuals = prices - filtered
    axes[1].plot(residuals, color=C_ANOMALY, lw=0.8, alpha=0.7)
    axes[1].axhline(0, color="black", ls="--", lw=1)
    axes[1].fill_between(range(len(residuals)), residuals, alpha=0.2, color=C_ANOMALY)
    axes[1].set_title("Залишки фільтра", fontsize=12)
    axes[1].set_xlabel("Індекс спостереження")
    axes[1].set_ylabel("Залишок, USD")

    fig.tight_layout()
    _save(fig, filename)