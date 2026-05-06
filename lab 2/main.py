# ============================================================
#   Лабораторна робота №2
#   ETF SPY (S&P 500) — 2015–2019
#   Група вимог 1: МНК + прогнозування
#   Група вимог 2: α-β фільтр
# ============================================================

import requests
import pandas as pd
import numpy as np
import math as mt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ============================================================
# ПАРСИНГ (аналогічно Лр_1, тільки тікер SPY)
# ============================================================

def parse_spy_prices() -> pd.DataFrame:
    """
    Парсинг цін ETF SPY (S&P 500).
    Джерело: https://finance.yahoo.com/quote/SPY/
    Той самий часовий проміжок що в Лр_1: 2015–2019.
    """
    ticker = "SPY"
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&period1=1420070400&period2=1577836800"
    )
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    print("=" * 60)
    print("- Парсинг сайту")
    print(f"Тікер: {ticker}")
    print(f"Джерело: https://finance.yahoo.com/quote/SPY/")

    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"HTTP статус: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            result = data["chart"]["result"][0]
            dates = pd.to_datetime(result["timestamp"], unit="s").normalize()
            q = result["indicators"]["quote"][0]
            df = pd.DataFrame({
                "Date"  : dates,
                "Close" : q["close"],
                "Open"  : q["open"],
                "High"  : q["high"],
                "Low"   : q["low"],
                "Volume": q["volume"],
            })
            df.dropna(subset=["Close"], inplace=True)
            df.sort_values("Date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            print(f"Рядків: {len(df)}")
            print(f"Період: {df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}")
            return df
    except Exception as e:
        print(f"!Мережа недоступна ({e})!")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df


def save_to_csv(df: pd.DataFrame, filename: str = "spy_prices.csv") -> None:
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print("=" * 60)
    print("- Збереження даних")
    print(f"Файл збережено: {os.path.abspath(filename)}")
    print(f"Рядків: {len(df)}")


# ============================================================
# МНК-ФУНКЦІЇ (з лекції L_1_3)
# ============================================================

def MNK_Stat_characteristics(S0):
    """МНК згладжування для визначення стат. характеристик (з лекції)."""
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F   = np.ones((iter, 3))
    for i in range(iter):
        Yin[i, 0] = float(S0[i])
        F[i, 1]   = float(i)
        F[i, 2]   = float(i * i)
    FT    = F.T
    C     = np.linalg.inv(FT.dot(F)).dot(FT).dot(Yin)
    Yout  = F.dot(C)
    return Yout


def MNK(S0):
    """МНК згладжування (з лекції L_1_3)."""
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F   = np.ones((iter, 3))
    for i in range(iter):
        Yin[i, 0] = float(S0[i])
        F[i, 1]   = float(i)
        F[i, 2]   = float(i * i)
    FT   = F.T
    C    = np.linalg.inv(FT.dot(F)).dot(FT).dot(Yin)
    Yout = F.dot(C)
    print("Регресійна модель:")
    print(f"y(t) = {C[0,0]:.6f}  +  {C[1,0]:.6f} * t  +  {C[2,0]:.8f} * t^2")
    return Yout


def MNK_AV_Detect(S0):
    """Детекція напрямку тренду (лінійний коефіцієнт МНК, з лекції)."""
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F   = np.ones((iter, 3))
    for i in range(iter):
        Yin[i, 0] = float(S0[i])
        F[i, 1]   = float(i)
        F[i, 2]   = float(i * i)
    FT = F.T
    C  = np.linalg.inv(FT.dot(F)).dot(FT).dot(Yin)
    return float(C[1, 0])


def MNK_Extrapol(S0, koef):
    """МНК прогнозування (з лекції L_1_3)."""
    iter           = len(S0)
    Yout_Extrapol  = np.zeros((iter + koef, 1))
    Yin = np.zeros((iter, 1))
    F   = np.ones((iter, 3))
    for i in range(iter):
        Yin[i, 0] = float(S0[i])
        F[i, 1]   = float(i)
        F[i, 2]   = float(i * i)
    FT = F.T
    C  = np.linalg.inv(FT.dot(F)).dot(FT).dot(Yin)
    print("Регресійна модель (прогнозування):")
    print(f"y(t) = {C[0,0]:.4f} + {C[1,0]:.6f} * t + {C[2,0]:.8f} * t^2")
    for i in range(iter + koef):
        Yout_Extrapol[i, 0] = C[0,0] + C[1,0]*i + C[2,0]*i*i
    return Yout_Extrapol


def r2_score(SL, Yout, Text):
    """Коефіцієнт детермінації (з лекції L_1_3)."""
    iter         = len(Yout)
    numerator    = 0
    denominator_1= 0
    for i in range(iter):
        numerator     += (SL[i] - Yout[i, 0]) ** 2
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
    """Статистичні характеристики вхідної вибірки (з лекції L_1_3)."""
    Yout = MNK_Stat_characteristics(SL)
    iter = len(Yout)
    SL0  = np.zeros(iter)
    for i in range(iter):
        SL0[i] = SL[i] - Yout[i, 0]
    mS   = np.median(SL0)
    dS   = np.var(SL0)
    scvS = mt.sqrt(dS)
    print(f"------------ {Text} -------------")
    print(f"Кількість елементів вибірки = {iter}")
    print(f"Математичне сподівання ВВ = {mS:.4f}")
    print(f"Дисперсія ВВ = {dS:.4f}")
    print(f"СКВ ВВ = {scvS:.4f}")
    print("-----------------------------------------------------")


def Stat_characteristics_out(SL_in, SL, Text):
    """Статистичні характеристики лінії тренду (з лекції L_1_3)."""
    Yout = MNK_Stat_characteristics(SL[:, 0])
    iter = len(Yout)
    SL0  = np.zeros(iter)
    for i in range(iter):
        SL0[i] = SL[i, 0] - Yout[i, 0]
    mS   = np.median(SL0)
    dS   = np.var(SL0)
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
    """Статистичні характеристики екстраполяції (з лекції L_1_3)."""
    Yout = MNK_Stat_characteristics(SL[:, 0])
    iter = len(Yout)
    SL0  = np.zeros(iter)
    for i in range(iter):
        SL0[i] = SL[i, 0] - Yout[i, 0]
    mS         = np.median(SL0)
    dS         = np.var(SL0)
    scvS       = mt.sqrt(dS)
    scvS_extrapol = scvS * koef
    print(f"------------ {Text} -------------")
    print(f"Кількість елементів вибірки = {iter}")
    print(f"Математичне сподівання ВВ = {mS:.4f}")
    print(f"Дисперсія ВВ = {dS:.4f}")
    print(f"СКВ ВВ = {scvS:.4f}")
    print(f"Довірчий інтервал прогнозу за СКВ = {scvS_extrapol:.4f}")
    print("-----------------------------------------------------")


# ============================================================
# ВИЯВЛЕННЯ АВ: МЕТОД MEDIUM (з лекції L_1_3)
# ============================================================

def Sliding_Window_AV_Detect_medium(S0, n_Wind, Q):
    """
    Виявлення та очищення АВ методом medium (з лекції L_1_3).
    Порівнює СКВ ковзного вікна з еталонним вікном.
    Q — коефіцієнт виявлення АВ.
    """
    iter    = len(S0)
    j_Wind  = mt.ceil(iter - n_Wind) + 1
    S0_Wind = np.zeros(n_Wind)

    # еталон (перше вікно)
    for i in range(n_Wind):
        S0_Wind[i]   = S0[i]
    dS_standart  = np.var(S0_Wind)
    scvS_standart= mt.sqrt(dS_standart)

    # ковзне вікно
    for j in range(j_Wind):
        for i in range(n_Wind):
            l = j + i
            S0_Wind[i] = S0[l]
        mS   = np.median(S0_Wind)
        dS   = np.var(S0_Wind)
        scvS = mt.sqrt(dS)
        # детекція та заміна АВ
        if scvS > (Q * scvS_standart):
            S0[l] = mS
    return S0


# ============================================================
# МОДЕЛЬ АВ (з лекції L_1_3, адаптовано для реальних даних)
# ============================================================

def Model_NORM_AV(S0_real, nAV_pct=5, Q_AV=3, seed=42):
    """
    Вносить аномальні виміри до реальних даних.
    nAV_pct — відсоток АВ від обсягу вибірки.
    Q_AV    — коефіцієнт переваги АВ (кратність СКВ).
    """
    rng  = np.random.default_rng(seed)
    n    = len(S0_real)
    nAV  = mt.ceil(n * nAV_pct / 100)
    dsig = float(np.std(S0_real))
    dm   = 0

    SV_AV = S0_real.copy()
    # рівномірно розподілені номери АВ (як у лекції)
    SAV   = [int(rng.integers(1, n)) for _ in range(nAV)]
    SSAV  = rng.normal(dm, Q_AV * dsig, nAV)  # аномальна похибка
    for i in range(nAV):
        k        = SAV[i]
        SV_AV[k] = S0_real[k] + SSAV[i]

    print("=" * 60)
    print("Вр.1/Вр.2 — Модель вхідних даних з АВ:")
    print(f"  Кількість АВ: {nAV} ({nAV_pct}% вибірки)")
    print(f"  Коефіцієнт переваги АВ Q = {Q_AV}")
    print(f"  СКВ похибки: {Q_AV*dsig:.2f} USD")
    return SV_AV, SAV


# ============================================================
# ПОКАЗНИКИ ЯКОСТІ (з лекції L_1_3)
# ============================================================

def evaluate_quality(S_clean, label):
    """
    Визначення показників якості моделі після очищення.
    Порівнює МНК-моделі для вибірки без АВ та після очищення.
    Показник якості: R² та динамічна похибка (Delta).
    """
    print("=" * 60)
    print(f"Вр.1/Вр.2 — Показники якості після очищення: {label}")
    Yout = MNK(S_clean)
    R2   = r2_score(S_clean, Yout, "МНК після очищення від АВ")
    Stat_characteristics_out(S_clean, Yout, "Стат. характеристики МНК після очищення")
    return Yout, R2


# ============================================================
# α-β ФІЛЬТР (з лекції L_1_4)
# ============================================================

def ABF(S0):
    """
    Рекурентний α-β фільтр (алгоритм з лекції L_1_4).
    Адаптивні коефіцієнти: alfa та beta змінюються на кожному кроці.
    Захист від розбіжності: обмеження alfa у межах (0, 1].
    """
    iter   = len(S0)
    Yin    = np.zeros((iter, 1))
    YoutAB = np.zeros((iter, 1))
    T0     = 1
    for i in range(iter):
        Yin[i, 0] = float(S0[i])

    # початкові дані для запуску фільтра (з лекції)
    Yspeed_retro = (Yin[1, 0] - Yin[0, 0]) / T0
    Yextra       = Yin[0, 0] + Yspeed_retro
    alfa         = 2 * (2*1 - 1) / (1 * (1 + 1))
    beta         = (6 / 1) * (1 + 1)
    YoutAB[0, 0] = Yin[0, 0] + alfa * (Yin[0, 0])

    # рекурентний прохід по вимірам (з лекції)
    for i in range(1, iter):
        innov        = Yin[i, 0] - Yextra
        # захист від розбіжності: обмеження інновації на 5σ
        sigma_loc    = float(np.std(Yin[max(0,i-10):i+1, 0]))
        if sigma_loc > 1e-9 and abs(innov) > 5 * sigma_loc:
            innov = np.sign(innov) * 5 * sigma_loc
        YoutAB[i, 0] = Yextra + alfa * innov
        Yspeed       = Yspeed_retro + (beta / T0) * innov
        Yspeed_retro = Yspeed
        Yextra       = YoutAB[i, 0] + Yspeed_retro
        # адаптивні коефіцієнти (з лекції) — захист від розбіжності alfa
        alfa = min(1.0, (2 * (2*i - 1)) / (i * (i + 1)))
        beta = min(1.0,  6 / (i * (i + 1)))

    return YoutAB


# ============================================================
# ГРАФІКИ
# ============================================================

def plot_trend(prices, Yout, title, filename):
    plt.figure(figsize=(12, 5))
    plt.plot(prices, color="steelblue", linewidth=1, label="Реальні ціни SPY")
    plt.plot(Yout[:, 0], color="crimson", linewidth=2, linestyle="--",
             label="МНК-тренд (квадратична модель)")
    plt.title(title, fontsize=13)
    plt.xlabel("Індекс спостереження")
    plt.ylabel("Ціна закриття, USD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Графік збережено: {os.path.abspath(filename)}")


def plot_av(S0, SV, title, filename):
    plt.figure(figsize=(12, 5))
    plt.plot(SV,  color="gray",      linewidth=0.8, alpha=0.7, label="З аномаліями")
    plt.plot(S0,  color="steelblue", linewidth=1.2, label="Оригінал")
    plt.title(title, fontsize=13)
    plt.xlabel("Індекс спостереження")
    plt.ylabel("Ціна, USD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Графік збережено: {os.path.abspath(filename)}")


def plot_mnk_extrapol(prices_clean, Yout_extrapol, n, koef, filename):
    plt.figure(figsize=(13, 5))
    x_hist = np.arange(n)
    x_full = np.arange(n + koef)
    plt.plot(x_hist, prices_clean, color="steelblue", linewidth=1,
             label="Дані (очищені від АВ)")
    plt.plot(x_hist, Yout_extrapol[:n, 0], color="crimson", linewidth=2,
             linestyle="--", label="МНК-тренд")
    plt.plot(x_full[n:], Yout_extrapol[n:, 0], color="orange", linewidth=2,
             linestyle=":", label=f"МНК-прогноз (+{koef} точок, 0.5 вибірки)")
    plt.axvline(n - 1, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    plt.title("Вр.1 — МНК: тренд та прогноз цін ETF SPY", fontsize=13)
    plt.xlabel("Індекс спостереження")
    plt.ylabel("Ціна, USD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Графік збережено: {os.path.abspath(filename)}")


def plot_abf(prices_av, prices_clean, YoutAB, filename):
    plt.figure(figsize=(13, 5))
    plt.plot(prices_av,    color="gray",    linewidth=0.7, alpha=0.6, label="З аномаліями (вхід фільтра)")
    plt.plot(prices_clean, color="steelblue", linewidth=1, alpha=0.8, label="Оригінал (еталон)")
    plt.plot(YoutAB[:, 0], color="crimson", linewidth=1.8, label="α-β фільтр (вихід)")
    plt.title("Вр.2 — α-β фільтр: рекурентне згладжування ETF SPY", fontsize=13)
    plt.xlabel("Індекс спостереження")
    plt.ylabel("Ціна, USD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Графік збережено: {os.path.abspath(filename)}")


def plot_histogram(prices, title, filename):
    Yout = MNK_Stat_characteristics(prices)
    residuals = np.array([prices[i] - Yout[i, 0] for i in range(len(prices))])
    m_mean = float(np.mean(residuals))
    plt.figure(figsize=(7, 4))
    plt.hist(residuals, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
    plt.axvline(m_mean, color="crimson", linestyle="--", label=f"mean={m_mean:.2f}")
    plt.title(title, fontsize=12)
    plt.xlabel("Відхилення від МНК-тренду, USD")
    plt.ylabel("Частота")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Гістограму збережено: {os.path.abspath(filename)}")


# ============================================================
# ГОЛОВНИЙ БЛОК
# ============================================================

if __name__ == "__main__":

    # ── Парсинг ──────────────────────────────────────────────
    df_raw   = parse_spy_prices()
    df_clean = clean_dataframe(df_raw)
    save_to_csv(df_clean, "spy_prices.csv")

    prices = df_clean["Close"].values.astype(float)
    n      = len(prices)

    # ── Стат. характеристики вихідної вибірки (Лр_1) ─────────
    print("=" * 60)
    print("- Оцінка динаміки тренду")
    total_change = prices[-1] - prices[0]
    pct_change   = (total_change / prices[0]) * 100
    print(f"Перша ціна: {prices[0]:.2f} USD")
    print(f"Остання ціна: {prices[-1]:.2f} USD")
    print(f"Загальна зміна: {total_change:+.2f} USD ({pct_change:+.2f}%)")

    Yout_trend = MNK_Stat_characteristics(prices)
    plot_trend(prices, Yout_trend, "Динаміка цін ETF SPY та МНК-тренд", "spy_trend.png")

    print("=" * 60)
    print("- Статистичні характеристики: ETF SPY (ціна закриття)")
    Stat_characteristics_in(prices, "Вихідна вибірка SPY (без АВ)")
    plot_histogram(prices, "Гістограма залишків: ETF SPY", "spy_histogram.png")

    # ════════════════════════════════════════════════════════
    # ГРУПА ВИМОГ 1
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ГРУПА ВИМОГ 1")
    print("=" * 60)

    # Вр.1 п.2 — Модель з АВ
    SV_AV_1, SAV_1 = Model_NORM_AV(prices, nAV_pct=5, Q_AV=3, seed=42)
    plot_av(prices, SV_AV_1, "Вр.1 — ETF SPY: вихідні дані + АВ", "vr1_av_model.png")
    Stat_characteristics_in(SV_AV_1, "Вр.1 — Вибірка з АВ")

    # Вр.1 п.3 — Очищення методом medium (з лекції L_1_3)
    print("=" * 60)
    print("Вр.1 п.3 — Очищення від АВ: метод medium (з лекції)")
    N_Wind_AV = 5
    Q_medium  = 1.6
    S_clean_1 = Sliding_Window_AV_Detect_medium(SV_AV_1.copy(), N_Wind_AV, Q_medium)
    Stat_characteristics_in(S_clean_1, "Вр.1 — Вибірка очищена від АВ (medium)")
    plot_av(prices, S_clean_1, "Вр.1 — ETF SPY: після очищення medium", "vr1_cleaned.png")

    # Вр.1 п.4 — Показники якості
    Yout_clean_1, R2_1 = evaluate_quality(S_clean_1, "Вр.1 очищена вибірка")

    # Вр.1 п.5 — МНК (поліноміальна регресія, з лекції)
    print("=" * 60)
    print("Вр.1 п.5 — Статистичне навчання МНК (поліноміальна регресія)")
    Yout_mnk_1 = MNK(S_clean_1)
    r2_score(S_clean_1, Yout_mnk_1, "Вр.1 МНК-модель")
    plot_trend(S_clean_1, Yout_mnk_1,
               "Вр.1 — МНК-згладжування ETF SPY (після очищення)", "vr1_mnk.png")

    # Вр.1 п.6 — Прогнозування на 0.5 інтервалу вибірки (з лекції)
    print("=" * 60)
    print("Вр.1 п.6 — МНК-прогнозування (екстраполяція на 0.5 вибірки)")
    koef_Extrapol = 0.5
    koef          = mt.ceil(n * koef_Extrapol)
    Yout_extrapol_1 = MNK_Extrapol(S_clean_1, koef)
    Stat_characteristics_extrapol(koef, Yout_extrapol_1, "Вр.1 МНК-прогнозування")
    plot_mnk_extrapol(S_clean_1, Yout_extrapol_1, n, koef, "vr1_forecast.png")

    # Вр.1 п.7 — Верифікація
    print("=" * 60)
    print("Вр.1 п.7 — Аналіз та верифікація")
    print(f"  R² МНК-моделі = {R2_1:.4f}")
    print(f"  Прогноз на {koef} точок (0.5 вибірки)")
    last_real     = prices[-1]
    last_forecast = Yout_extrapol_1[-1, 0]
    print(f"  Остання реальна ціна: {last_real:.2f} USD")
    print(f"  Прогнозована ціна через {koef} кроків: {last_forecast:.2f} USD")

    # ════════════════════════════════════════════════════════
    # ГРУПА ВИМОГ 2
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ГРУПА ВИМОГ 2")
    print("=" * 60)

    # Вр.2 п.2 — Та сама модель з АВ (інший seed для різноманітності)
    SV_AV_2, SAV_2 = Model_NORM_AV(prices, nAV_pct=5, Q_AV=3, seed=99)
    plot_av(prices, SV_AV_2, "Вр.2 — ETF SPY: вихідні дані + АВ", "vr2_av_model.png")
    Stat_characteristics_in(SV_AV_2, "Вр.2 — Вибірка з АВ")

    # Вр.2 п.3 — Очищення (sliding window, з лекції L_1_3)
    print("=" * 60)
    print("Вр.2 п.3 — Очищення від АВ: sliding window (з лекції)")
    # Реалізація sliding window з лекції (Sliding_Window_AV_Detect_sliding_wind)
    def Sliding_Window_AV_Detect_sliding_wind(S0, n_Wind):
        iter    = len(S0)
        j_Wind  = mt.ceil(iter - n_Wind) + 1
        S0_Wind = np.zeros(n_Wind)
        Midi    = np.zeros(iter)
        for j in range(j_Wind):
            for i in range(n_Wind):
                l          = j + i
                S0_Wind[i] = S0[l]
            Midi[l] = np.median(S0_Wind)
        S0_Midi = np.zeros(iter)
        for j in range(iter):
            S0_Midi[j] = Midi[j]
        for j in range(n_Wind):
            S0_Midi[j] = S0[j]
        return S0_Midi

    n_Wind    = 5
    S_clean_2 = Sliding_Window_AV_Detect_sliding_wind(SV_AV_2.copy(), n_Wind)
    Stat_characteristics_in(S_clean_2, "Вр.2 — Вибірка очищена від АВ (sliding window)")
    plot_av(prices, S_clean_2, "Вр.2 — ETF SPY: після очищення sliding window", "vr2_cleaned.png")

    # Вр.2 п.4 — Показники якості
    Yout_clean_2, R2_2 = evaluate_quality(S_clean_2, "Вр.2 очищена вибірка")
    print(f"  R² = {R2_2:.4f} → {'α-β' if R2_2 < 0.7 else 'α-β'} фільтр (обирається завжди для Вр.2)")

    # Вр.2 п.5 — α-β фільтр (з лекції L_1_4)
    print("=" * 60)
    print("Вр.2 п.5 — α-β фільтр (рекурентне згладжування, з лекції L_1_4)")
    print("  Захист від розбіжності: clipping інновації на рівні 5σ")
    YoutAB = ABF(SV_AV_2)
    r2_score_abf = r2_score(S_clean_2, YoutAB, "Вр.2 α-β фільтр")
    rmse_before  = mt.sqrt(float(np.mean((SV_AV_2 - prices)**2)))
    rmse_after   = mt.sqrt(float(np.mean((YoutAB[:, 0] - prices)**2)))
    print(f"  RMSE до фільтрації  : {rmse_before:.4f} USD")
    print(f"  RMSE після фільтрації: {rmse_after:.4f} USD")
    print(f"  Покращення           : {(1 - rmse_after/rmse_before)*100:.1f}%")
    plot_abf(SV_AV_2, prices, YoutAB, "vr2_abf.png")

    # Вр.2 п.6 — Аналіз та верифікація
    print("=" * 60)
    print("Вр.2 п.6 — Аналіз та верифікація")
    Stat_characteristics_out(S_clean_2, YoutAB, "Вр.2 α-β фільтр — статистичні характеристики")

    print("=" * 60)
    print("Лабораторну роботу №2 виконано.")
    print("Вихідні файли:")
    files = [
        "spy_prices.csv", "spy_trend.png", "spy_histogram.png",
        "vr1_av_model.png", "vr1_cleaned.png", "vr1_mnk.png", "vr1_forecast.png",
        "vr2_av_model.png", "vr2_cleaned.png", "vr2_abf.png",
    ]
    for f in files:
        print(f"  {f}")