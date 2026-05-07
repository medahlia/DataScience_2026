# ============================================================
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

from mnk import (MNK, MNK_Stat_characteristics, MNK_Extrapol,
                 r2_score, Stat_characteristics_in,
                 Stat_characteristics_out, Stat_characteristics_extrapol)
from anomaly import (Model_NORM_AV, Sliding_Window_AV_Detect_medium,
                     Sliding_Window_AV_Detect_sliding_wind)
from abf import ABF


def parse_spy_prices() -> pd.DataFrame:
    """
    Парсинг цін ETF SPY (S&P 500).
    Джерело: https://finance.yahoo.com/quote/SPY/
    часовий проміжок: 2015–2019.
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


# графіки
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


if __name__ == "__main__":

    # парсинг та збереження
    df = parse_spy_prices()
    save_to_csv(df, "spy_prices.csv")
    prices = df["Close"].values.astype(float)
    n = len(prices)

    # стат. характеристики вихідної вибірки
    print("=" * 60)
    print("- Оцінка динаміки тренду")
    total_change = prices[-1] - prices[0]
    pct_change = (total_change / prices[0]) * 100
    print(f"Перша ціна : {prices[0]:.2f} USD")
    print(f"Остання ціна: {prices[-1]:.2f} USD")
    print(f"Загальна зміна: {total_change:+.2f} USD ({pct_change:+.2f}%)")

    Yout_trend = MNK_Stat_characteristics(prices)
    plot_trend(prices, Yout_trend, "Динаміка цін ETF SPY та МНК-тренд", "spy_trend.png")

    print("=" * 60)
    print("- Статистичні характеристики: ETF SPY (ціна закриття)")
    Stat_characteristics_in(prices, "Вихідна вибірка SPY (без АВ)")
    plot_histogram(prices, "Гістограма залишків: ETF SPY", "spy_histogram.png")

    # ГРУПА ВИМОГ 1

    print("\n" + "=" * 60)
    print("  ГРУПА ВИМОГ 1")
    print("=" * 60)

    # п.2 Модель з АВ
    SV_AV_1, _ = Model_NORM_AV(prices, nAV_pct=5, Q_AV=3, seed=42)
    plot_av(prices, SV_AV_1, "Вр.1 | ETF SPY: вихідні дані + АВ", "vr1_av_model.png")
    Stat_characteristics_in(SV_AV_1, "Вр.1 | Вибірка з АВ")

    # п.3 Очищення методом medium (L_1_3)
    print("=" * 60)
    print("Вр.1 | Очищення від АВ: метод medium")
    N_Wind_AV = 5
    Q_medium = 1.6
    S_clean_1 = Sliding_Window_AV_Detect_medium(SV_AV_1.copy(), N_Wind_AV, Q_medium)
    Stat_characteristics_in(S_clean_1, "Вр.1 | Вибірка очищена від АВ (medium)")
    plot_av(prices, S_clean_1, "Вр.1 | ETF SPY: після очищення medium", "vr1_cleaned.png")

    # п.4 Показники якості
    print("=" * 60)
    print("Вр.1 | Показники якості моделі після очищення")
    Yout_clean_1 = MNK(S_clean_1)
    R2_1 = r2_score(S_clean_1, Yout_clean_1, "Вр.1 | МНК після очищення від АВ")
    Stat_characteristics_out(S_clean_1, Yout_clean_1, "Вр.1 | стат. характеристики МНК")

    # п.5 МНК-навчання (L_1_3)
    print("=" * 60)
    print("Вр.1 | Статистичне навчання МНК (поліноміальна регресія)")
    plot_trend(S_clean_1, Yout_clean_1,
               "Вр.1 | МНК-згладжування ETF SPY (після очищення)", "vr1_mnk.png")

    # п.6 прогнозування на 0.5 інтервалу вибірки (L_1_3)
    print("=" * 60)
    print("Вр.1 | МНК-прогнозування (екстраполяція на 0.5 вибірки)")
    koef = mt.ceil(n * 0.5)
    Yout_extrapol_1 = MNK_Extrapol(S_clean_1, koef)
    Stat_characteristics_extrapol(koef, Yout_extrapol_1, "Вр.1 | МНК-прогнозування")
    plot_mnk_extrapol(S_clean_1, Yout_extrapol_1, n, koef, "vr1_forecast.png")

    # п.7 Аналіз та верифікація
    print("=" * 60)
    print("Вр.1 |  Аналіз та верифікація")
    print(f"R² МНК-моделі = {R2_1:.4f}")
    print(f"Прогноз на {koef} точок (0.5 вибірки)")
    print(f"Остання реальна ціна: {prices[-1]:.2f} USD")
    print(f"Прогнозована ціна через {koef} кроків: {Yout_extrapol_1[-1, 0]:.2f} USD")

    # ГРУПА ВИМОГ 2

    print("\n" + "=" * 60)
    print("  ГРУПА ВИМОГ 2")
    print("=" * 60)

    # п.2 Модель з АВ (змінюємо seed)
    SV_AV_2, _ = Model_NORM_AV(prices, nAV_pct=5, Q_AV=3, seed=99)
    plot_av(prices, SV_AV_2, "Вр.2 | ETF SPY: вихідні дані + АВ", "vr2_av_model.png")
    Stat_characteristics_in(SV_AV_2, "Вр.2 | Вибірка з АВ")

    # п.3 очищення sliding window (L_1_3)
    print("=" * 60)
    print("Вр.2 | Очищення від АВ: sliding window (L_1_3)")
    n_Wind = 5
    S_clean_2 = Sliding_Window_AV_Detect_sliding_wind(SV_AV_2.copy(), n_Wind)
    Stat_characteristics_in(S_clean_2, "Вр.2 | Вибірка очищена від АВ (sliding window)")
    plot_av(prices, S_clean_2, "Вр.2 | ETF SPY: після очищення sliding window", "vr2_cleaned.png")

    # п.4 Показники якості
    print("=" * 60)
    print("Вр.2 | Показники якості моделі після очищення")
    Yout_clean_2 = MNK(S_clean_2)
    R2_2 = r2_score(S_clean_2, Yout_clean_2, "Вр.2 | МНК після очищення від АВ")
    Stat_characteristics_out(S_clean_2, Yout_clean_2, "Вр.2 | стат. характеристики МНК")
    print(f"  R² = {R2_2:.4f} → застосовується α-β фільтр")

    # п.5 α-β фільтр (L_1_4)
    print("=" * 60)
    print("Вр.2 | α-β фільтр (рекурентне згладжування)")
    YoutAB = ABF(SV_AV_2)
    r2_score(S_clean_2, YoutAB, "Вр.2 | α-β фільтр")
    rmse_before = mt.sqrt(float(np.mean((SV_AV_2 - prices) ** 2)))
    rmse_after = mt.sqrt(float(np.mean((YoutAB[:, 0] - prices) ** 2)))
    print(f"  RMSE до фільтрації: {rmse_before:.4f} USD")
    print(f"  RMSE після фільтрації: {rmse_after:.4f} USD")
    print(f"  Покращення: {(1 - rmse_after / rmse_before) * 100:.1f}%")
    plot_abf(SV_AV_2, prices, YoutAB, "vr2_abf.png")

    # п.6 Аналіз та верифікація
    print("=" * 60)
    print("Вр.2 | Аналіз та верифікація")
    Stat_characteristics_out(S_clean_2, YoutAB,"Вр.2 | α-β фільтр - статистичні характеристики")

    print("=" * 60)