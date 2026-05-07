# Функції генерації АВ та очищення

import numpy as np
import math as mt


def Model_NORM_AV(S0_real, nAV_pct=5, Q_AV=3, seed=42):
    """
    Модель вхідних даних з аномальними вимірами (аналог з лекції L_1_3).
    nAV_pct — відсоток АВ від обсягу вибірки.
    Q_AV    — коефіцієнт переваги АВ (кратність СКВ).
    Номери АВ розподілені рівномірно (як у лекції: randomAM + Model_NORM_AV).
    """
    rng = np.random.default_rng(seed)
    n = len(S0_real)
    nAV = mt.ceil(n * nAV_pct / 100)
    dsig = float(np.std(S0_real))
    dm = 0

    SV_AV = S0_real.copy()
    # рівномірно розподілені номери АВ в межах вибірки
    SAV = [int(rng.integers(1, n)) for _ in range(nAV)]
    # аномальна випадкова похибка з нормальним законом
    SSAV = rng.normal(dm, Q_AV * dsig, nAV)
    for i in range(nAV):
        k = SAV[i]
        SV_AV[k] = S0_real[k] + SSAV[i]

    print("=" * 60)
    print("Модель вхідних даних з АВ:")
    print(f"  Кількість АВ: {nAV} ({nAV_pct}% вибірки)")
    print(f"  Коефіцієнт переваги АВ Q = {Q_AV}")
    print(f"  СКВ аномальної похибки: {Q_AV * dsig:.2f} USD")
    return SV_AV, SAV


def Sliding_Window_AV_Detect_medium(S0, n_Wind, Q):
    """
    Виявлення та очищення АВ методом medium (L_1_3).
    Порівнює СКВ ковзного вікна з еталонним (першим) вікном.
    Q — коефіцієнт виявлення АВ.
    """
    iter = len(S0)
    j_Wind = mt.ceil(iter - n_Wind) + 1
    S0_Wind = np.zeros(n_Wind)

    # еталон — перше вікно
    for i in range(n_Wind):
        S0_Wind[i] = S0[i]
    dS_standart = np.var(S0_Wind)
    scvS_standart = mt.sqrt(dS_standart)

    # ковзне вікно
    for j in range(j_Wind):
        for i in range(n_Wind):
            l = j + i
            S0_Wind[i] = S0[l]
        mS = np.median(S0_Wind)
        dS = np.var(S0_Wind)
        scvS = mt.sqrt(dS)
        # детекція та заміна АВ медіаною вікна
        if scvS > (Q * scvS_standart):
            S0[l] = mS
    return S0


def Sliding_Window_AV_Detect_sliding_wind(S0, n_Wind):
    """
    Виявлення та очищення АВ методом sliding window (L_1_3)
    Замінює кожну точку медіаною ковзного вікна
    """
    iter = len(S0)
    j_Wind = mt.ceil(iter - n_Wind) + 1
    S0_Wind = np.zeros(n_Wind)
    Midi = np.zeros(iter)

    # ковзне вікно
    for j in range(j_Wind):
        for i in range(n_Wind):
            l = j + i
            S0_Wind[i] = S0[l]
        Midi[l] = np.median(S0_Wind)

    # формування очищеної вибірки
    S0_Midi = np.zeros(iter)
    for j in range(iter):
        S0_Midi[j] = Midi[j]
    for j in range(n_Wind):
        S0_Midi[j] = S0[j]
    return S0_Midi