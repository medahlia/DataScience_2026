"""
mnk.py — Методи найменших квадратів (МНК / LSM).
Поліноміальна регресія ступенів 1–4, екстраполяція.
"""
import math as mt
import numpy as np


#????????????
def _build_vandermonde(n: int, degree: int) -> np.ndarray:
    """Матриця Вандермонда розміром (n, degree+1)."""
    F = np.ones((n, degree + 1))
    for i in range(n):
        for d in range(1, degree + 1):
            F[i, d] = float(i ** d)
    return F


def mnk_poly(S0: np.ndarray, degree: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    МНК-апроксимація поліномом заданого ступеня.
    Повертає (Yout, C) — згладжений ряд та коефіцієнти.
    """
    n = len(S0)
    Yin = S0.reshape(-1, 1).astype(float)
    F = _build_vandermonde(n, degree)
    FT = F.T
    C = np.linalg.inv(FT @ F) @ FT @ Yin
    Yout = F @ C
    return Yout, C

"""
def mnk_extrapol(S0: np.ndarray, koef: int, degree: int = 2) -> np.ndarray:
    
    #МНК-екстраполяція на koef кроків вперед.
    #Повертає масив довжиною (n + koef, 1).
    
    n = len(S0)
    _, C = mnk_poly(S0, degree)
    total = n + koef
    Yout = np.zeros((total, 1))
    for i in range(total):
        val = 0.0
        for d in range(degree + 1):
            val += C[d, 0] * (i ** d)
        Yout[i, 0] = val
    return Yout
"""

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

"""
def r2_score_mnk(prices: np.ndarray, Yout: np.ndarray, label: str = "") -> float:
    n = len(prices)
    residuals = prices - Yout[:n, 0]
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((prices - np.mean(prices)) ** 2)
    R2 = 1 - ss_res / ss_tot
    if label:
        print(f"[{label}] R² = {R2:.6f}")
    return R2
"""


# ── Зворотна сумісність: оригінальні функції (квадратична, degree=2) ──────────

def MNK_Stat_characteristics(S0: np.ndarray) -> np.ndarray:
    Yout, _ = mnk_poly(S0, degree=2)
    return Yout


def MNK(S0: np.ndarray) -> np.ndarray:
    Yout, C = mnk_poly(S0, degree=2)
    print(f"Регресійна МНК-модель:")
    print(f"y(t) = {C[0, 0]:.6f}  +  {C[1, 0]:.6f} * t  +  {C[2, 0]:.8f} * t^2")
    return Yout


def MNK_AV_Detect(S0: np.ndarray) -> float:
    _, C = mnk_poly(S0, degree=2)
    return float(C[1, 0])


#???????????/////
def mnk_extrapol(S0: np.ndarray, koef: int, degree: int = 2) -> np.ndarray:
    n = len(S0)

    # отримуємо коефіцієнти
    _, C = mnk_poly(S0, degree)

    total = n + koef
    Yout = np.zeros((total, 1))

    for i in range(total):
        val = 0.0
        for d in range(degree + 1):
            val += C[d, 0] * (i ** d)
        Yout[i, 0] = val

    return Yout


def MNK_Extrapol(S0: np.ndarray, koef: int) -> np.ndarray:
    Yout = mnk_extrapol(S0, koef, degree=2)
    n = len(S0)
    _, C = mnk_poly(S0, degree=2)
    print(f"МНК-прогноз: y(t) = {C[0, 0]:.4f} + {C[1, 0]:.6f}*t + {C[2, 0]:.8f}*t^2")
    return Yout
