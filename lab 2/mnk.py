# МНК-функції

import numpy as np
import math as mt


def MNK_Stat_characteristics(S0):
    """МНК-згладжування для визначення статистичних характеристик"""
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):
        Yin[i, 0] = float(S0[i])
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    C = np.linalg.inv(FT.dot(F)).dot(FT).dot(Yin)
    Yout = F.dot(C)
    return Yout


def MNK(S0):
    """МНК-згладжування з виведенням коефіцієнтів моделі"""
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):
        Yin[i, 0] = float(S0[i])
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    C = np.linalg.inv(FT.dot(F)).dot(FT).dot(Yin)
    Yout = F.dot(C)
    print("Регресійна модель:")
    print(f"y(t) = {C[0,0]:.6f}  +  {C[1,0]:.6f} * t  +  {C[2,0]:.8f} * t^2")
    return Yout


def MNK_Extrapol(S0, koef):
    """МНК-прогнозування на koef кроків вперед (L_1_3)"""
    iter = len(S0)
    Yout_Extrapol = np.zeros((iter + koef, 1))
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):
        Yin[i, 0] = float(S0[i])
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    C = np.linalg.inv(FT.dot(F)).dot(FT).dot(Yin)
    print("Регресійна модель (прогнозування):")
    print(f"y(t) = {C[0,0]:.4f} + {C[1,0]:.6f} * t + {C[2,0]:.8f} * t^2")
    for i in range(iter + koef):
        Yout_Extrapol[i, 0] = C[0,0] + C[1,0]*i + C[2,0]*i*i
    return Yout_Extrapol


def r2_score(SL, Yout, Text):
    """Коефіцієнт детермінації R² (L_1_3)"""
    iter = len(Yout)
    numerator = 0
    denominator_1 = 0
    for i in range(iter):
        numerator += (SL[i] - Yout[i, 0]) ** 2
        denominator_1 += SL[i]
    denominator_2 = 0
    for i in range(iter):
        denominator_2 += (SL[i] - (denominator_1 / iter)) ** 2
    R2 = 1 - (numerator / denominator_2)
    print(f"------------ {Text} -------------")
    print(f"Кількість елементів вибірки = {iter}")
    print(f"Коефіцієнт детермінації R² = {R2:.6f}")
    return R2


def Stat_characteristics_in(SL, Text):
    """Статистичні характеристики вхідної вибірки"""
    Yout = MNK_Stat_characteristics(SL)
    iter = len(Yout)
    SL0 = np.zeros(iter)
    for i in range(iter):
        SL0[i] = SL[i] - Yout[i, 0]
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    print(f"------------ {Text} -------------")
    print(f"Кількість елементів вибірки = {iter}")
    print(f"Математичне сподівання ВВ = {mS:.4f}")
    print(f"Дисперсія ВВ = {dS:.4f}")
    print(f"СКВ ВВ = {scvS:.4f}")
    print("-----------------------------------------------------")


def Stat_characteristics_out(SL_in, SL, Text):
    """Статистичні характеристики лінії тренду"""
    Yout = MNK_Stat_characteristics(SL[:, 0])
    iter = len(Yout)
    SL0 = np.zeros(iter)
    for i in range(iter):
        SL0[i] = SL[i, 0] - Yout[i, 0]
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    Delta = sum(abs(SL_in[i] - Yout[i, 0]) for i in range(iter))
    Delta_avg = Delta / (iter + 1)
    print(f"------------ {Text} -------------")
    print(f"Кількість елементів вибірки = {iter}")
    print(f"Математичне сподівання ВВ = {mS:.4f}")
    print(f"Дисперсія ВВ = {dS:.4f}")
    print(f"СКВ ВВ = {scvS:.4f}")
    print(f"Динамічна похибка моделі = {Delta_avg:.4f}")
    print("-----------------------------------------------------")


def Stat_characteristics_extrapol(koef, SL, Text):
    """Статистичні характеристики екстраполяції"""
    Yout = MNK_Stat_characteristics(SL[:, 0])
    iter = len(Yout)
    SL0 = np.zeros(iter)
    for i in range(iter):
        SL0[i] = SL[i, 0] - Yout[i, 0]
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    scvS_extrapol = scvS * koef
    print(f"------------ {Text} -------------")
    print(f"Кількість елементів вибірки = {iter}")
    print(f"Математичне сподівання ВВ = {mS:.4f}")
    print(f"Дисперсія ВВ = {dS:.4f}")
    print(f"СКВ ВВ = {scvS:.4f}")
    print(f"Довірчий інтервал прогнозу за СКВ = {scvS_extrapol:.4f}")
    print("-----------------------------------------------------")