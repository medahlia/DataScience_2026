# α-β фільтр

import numpy as np


def ABF(S0):
    """ змінити !!!!!!!!!!!!!!!
    Рекурентний α-β фільтр (L_1_4)

    Адаптивні коефіцієнти alfa та beta змінюються на кожному кроці:
        alfa = 2*(2*i - 1) / (i*(i + 1))
        beta = 6 / (i*(i + 1))

    Захист від розбіжності (додано до лекційного коду):
        інновація обмежується на рівні 5σ локального вікна,
        що запобігає «розльоту» фільтра при різких АВ.
    """
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    YoutAB = np.zeros((iter, 1))
    T0 = 1

    for i in range(iter):
        Yin[i, 0] = float(S0[i])

    # початкові дані для запуску фільтра
    Yspeed_retro = (Yin[1, 0] - Yin[0, 0]) / T0
    Yextra = Yin[0, 0] + Yspeed_retro
    alfa = 2 * (2*1 - 1) / (1 * (1 + 1))
    beta = (6 / 1) * (1 + 1)
    YoutAB[0, 0] = Yin[0, 0] + alfa * Yin[0, 0]

    # рекурентний прохід по вимірах
    for i in range(1, iter):
        innov = Yin[i, 0] - Yextra

        # захист від розбіжності: clipping інновації на рівні 5σ
        sigma_loc = float(np.std(Yin[max(0, i-10):i+1, 0]))
        if sigma_loc > 1e-9 and abs(innov) > 5 * sigma_loc:
            innov = np.sign(innov) * 5 * sigma_loc

        YoutAB[i, 0] = Yextra + alfa * innov
        Yspeed = Yspeed_retro + (beta / T0) * innov
        Yspeed_retro = Yspeed
        Yextra = YoutAB[i, 0] + Yspeed_retro

        # адаптивні коефіцієнти, обмежені у (0, 1]
        alfa = min(1.0, (2 * (2*i - 1)) / (i * (i + 1)))
        beta = min(1.0,  6 / (i * (i + 1)))

    return YoutAB