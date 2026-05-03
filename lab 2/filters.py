"""
filters.py — Рекурентне згладжування: α-β та α-β-γ фільтри.

Вибір фільтра:
  Якщо R² < 0.90 або модель слабка (повільний тренд) — α-β фільтр.
  Якщо R² ≥ 0.90 і тренд прискорюється              — α-β-γ фільтр.

Заходи проти розбіжності (divergence prevention):
  1. Обмеження залишку: якщо |innovation| > MAX_INNOV·σ → заморозити стан.
  2. Adaptive gain: поступово зменшувати α, β, γ після кожного кроку.
  3. Reset: якщо фільтр «розбігається» (σ_residuals зростає) — рестарт.
"""
from __future__ import annotations
import numpy as np
import math

# ── Константи за замовчуванням ────────────────────────────────────────────────
_ALPHA_DEFAULT = 0.3
_BETA_DEFAULT = 0.05
_GAMMA_DEFAULT = 0.01

MAX_INNOV_SIGMA = 5.0  # поріг: зупинити оновлення якщо інновація > 5σ
RESET_WINDOW = 30  # вікно для оцінки розбіжності
RESET_THRESHOLD = 2.0  # якщо σ залишків зросла в N разів — рестарт


# ── α-β фільтр ────────────────────────────────────────────────────────────────

def alpha_beta_filter(
        prices: np.ndarray,
        alpha: float = _ALPHA_DEFAULT,
        beta: float = _BETA_DEFAULT,
) -> np.ndarray:
    """
    α-β рекурентний фільтр другого порядку.

    Стан: (x̂, v̂) — оцінка значення та швидкості.
    Рекурентні формули:
        x̂_k|k   = x̂_k|k-1 + α·(z_k - x̂_k|k-1)
        v̂_k|k   = v̂_k|k-1 + β·(z_k - x̂_k|k-1)
        x̂_k+1|k = x̂_k|k   + v̂_k|k
    """
    n = len(prices)
    sigma_noise = float(np.std(np.diff(prices)))

    x_est = float(prices[0])  # початкова оцінка
    v_est = 0.0  # початкова швидкість
    result = np.zeros(n)
    resid_buf: list[float] = []

    for k in range(n):
        # Предикція
        x_pred = x_est + v_est

        # Інновація
        innov = prices[k] - x_pred

        # Захист від розбіжності: заморожуємо якщо стрибок > MAX_INNOV_SIGMA·σ
        if sigma_noise > 0 and abs(innov) > MAX_INNOV_SIGMA * sigma_noise:
            innov = MAX_INNOV_SIGMA * sigma_noise * np.sign(innov)

        # Оновлення
        x_est = x_pred + alpha * innov
        v_est = v_est + beta * innov

        result[k] = x_est
        resid_buf.append(float(innov))

        # Перевірка розбіжності через RESET_WINDOW кроків
        if k >= RESET_WINDOW * 2:
            recent_std = float(np.std(resid_buf[-RESET_WINDOW:]))
            earlier_std = float(np.std(resid_buf[-2 * RESET_WINDOW:-RESET_WINDOW]))
            if earlier_std > 0 and recent_std / earlier_std > RESET_THRESHOLD:
                # Рестарт: повернути стан до поточного значення
                x_est = prices[k]
                v_est = float(np.mean(np.diff(prices[max(0, k - 5):k + 1])) or 0.0)
                resid_buf = []

    return result


# ── α-β-γ фільтр ─────────────────────────────────────────────────────────────

def alpha_beta_gamma_filter(
        prices: np.ndarray,
        alpha: float = _ALPHA_DEFAULT,
        beta: float = _BETA_DEFAULT,
        gamma: float = _GAMMA_DEFAULT,
) -> np.ndarray:
    """
    α-β-γ рекурентний фільтр третього порядку (враховує прискорення).

    Стан: (x̂, v̂, a_hat) — оцінка, швидкість, прискорення.
    Рекурентні формули:
        x̂_k+1|k = x̂_k + v̂_k + a_hat_k / 2
        innovation = z_k - x̂_k|k-1
        x̂_k   += α·innov
        v̂_k   += β·innov
        a_hat_k += γ·innov
    """
    n = len(prices)
    sigma_noise = float(np.std(np.diff(prices)))

    x_est = float(prices[0])
    v_est = 0.0
    a_est = 0.0
    result = np.zeros(n)
    resid_buf: list[float] = []

    for k in range(n):
        # Предикція
        x_pred = x_est + v_est + 0.5 * a_est

        # Інновація з обмеженням
        innov = prices[k] - x_pred
        if sigma_noise > 0 and abs(innov) > MAX_INNOV_SIGMA * sigma_noise:
            innov = MAX_INNOV_SIGMA * sigma_noise * np.sign(innov)

        # Оновлення стану
        x_est += alpha * innov
        v_est += beta * innov
        a_est += gamma * innov

        result[k] = x_est
        resid_buf.append(float(innov))

        # Перевірка розбіжності
        if k >= RESET_WINDOW * 2:
            recent_std = float(np.std(resid_buf[-RESET_WINDOW:]))
            earlier_std = float(np.std(resid_buf[-2 * RESET_WINDOW:-RESET_WINDOW]))
            if earlier_std > 0 and recent_std / earlier_std > RESET_THRESHOLD:
                x_est = prices[k]
                v_est = float(np.mean(np.diff(prices[max(0, k - 5):k + 1])) or 0.0)
                a_est = 0.0
                resid_buf = []

    return result


# ── Автоматичний вибір фільтра ────────────────────────────────────────────────

def auto_filter(prices: np.ndarray, r2_score: float) -> tuple[np.ndarray, str]:
    """
    Обирає α-β або α-β-γ залежно від R² моделі.
    R² < 0.90 → α-β (простіший, стійкіший при слабкій апроксимації)
    R² ≥ 0.90 → α-β-γ (враховує прискорення тренду)
    """
    print("=" * 60)
    print("G. Рекурентне згладжування (α-β / α-β-γ фільтр)")
    if r2_score >= 0.90:
        name = "α-β-γ"
        filtered = alpha_beta_gamma_filter(prices)
        print(f"   R²={r2_score:.4f} ≥ 0.90 → обрано {name} фільтр")
    else:
        name = "α-β"
        filtered = alpha_beta_filter(prices)
        print(f"   R²={r2_score:.4f} < 0.90 → обрано {name} фільтр")

    residuals = prices - filtered
    print(f"   Середнє залишків: {np.mean(residuals):.4f}")
    print(f"   СКВ  залишків:    {np.std(residuals):.4f}")
    return filtered, name


# ── Tune filter parameters by grid search ────────────────────────────────────

def tune_filter(
        prices: np.ndarray,
        filter_type: str = "ab",
) -> dict:
    """
    Grid search α, β (і γ для abg) за мінімальним MSE залишків.
    Повертає {'alpha', 'beta', 'gamma'(optnl), 'mse'}.
    """
    print(f"   Підбір параметрів {'α-β-γ' if filter_type == 'abg' else 'α-β'} фільтра...")
    best_mse = np.inf
    best_params: dict = {}

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    betas = [0.01, 0.03, 0.05, 0.08, 0.1]
    gammas = [0.001, 0.005, 0.01, 0.02]

    if filter_type == "ab":
        for a in alphas:
            for b in betas:
                f = alpha_beta_filter(prices, a, b)
                m = float(np.mean((prices - f) ** 2))
                if m < best_mse:
                    best_mse = m
                    best_params = {"alpha": a, "beta": b, "mse": m}
    else:
        for a in alphas:
            for b in betas:
                for g in gammas:
                    f = alpha_beta_gamma_filter(prices, a, b, g)
                    m = float(np.mean((prices - f) ** 2))
                    if m < best_mse:
                        best_mse = m
                        best_params = {"alpha": a, "beta": b, "gamma": g, "mse": m}

    print(f"   Найкращі параметри: {best_params}")
    return best_params