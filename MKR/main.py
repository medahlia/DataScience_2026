"""
Сформувати модель нормального закону та визначити статистичні характеристики вибірки
"""

import numpy as np
import math as mt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def stat_characteristics(S):
    mS = np.mean(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    print('------- статистичні характеристики НОРМАЛЬНОГО закону розподілу ВВ -----')
    print('кількість елементів вибірки =', len(S))
    print('математичне сподівання ВВ =', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ =', scvS)
    print('------------------------------------------------------------------------')


def rando_norm(dm, dsig, iter):
    S = np.random.normal(dm, dsig, iter)
    stat_characteristics(S)

    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.title("Нормальний закон розподілу ВВ")
    plt.xlabel("Значення")
    plt.ylabel("Частота")
    plt.savefig("normal_distribution.png", dpi=120)
    plt.close()
    return S


if __name__ == '__main__':
    dm = 0
    dsig = 5
    iter = 10000
    rando_norm(dm, dsig, iter)