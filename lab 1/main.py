import requests
import pandas as pd
import numpy as np
import math as mt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def parse_gold_prices() -> pd.DataFrame:
    """
    Парсинг цін ETF GLD.
    дані: https://finance.yahoo.com/quote/GLD/)
    """
    ticker = "GLD"
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&period1=1420070400&period2=1577836800" # period
    )
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    print("=" * 60)
    print("1. Парсинг сайту")
    print(f"Тікер: {ticker}")
    print(f"Джерело: https://finance.yahoo.com/quote/GLD/")

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


def save_to_csv(df: pd.DataFrame, filename: str = "gold_prices.csv") -> None:
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print("=" * 60)
    print("2. Збереження даних")
    print(f"Файл збережено: {os.path.abspath(filename)}")
    print(f"Рядків: {len(df)}")


def plot_trend(prices: np.ndarray, Yout: np.ndarray, title: str, filename: str) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(prices, color="steelblue", linewidth=1, label="Реальні ціни GLD")
    plt.plot(Yout[:, 0], color="crimson", linewidth=2, linestyle="--", label="МНК-тренд (квадратична модель)")
    plt.title(title, fontsize=13)
    plt.xlabel("Індекс спостереження")
    plt.ylabel("Ціна закриття, USD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Графік збережено: {os.path.abspath(filename)}")


def analyze_trend(prices: np.ndarray) -> None:
    print("=" * 60)
    print("3. Оцінка динаміки тренду")
    total_change = prices[-1] - prices[0]
    pct_change = (total_change / prices[0]) * 100
    n = len(prices)
    avg_daily_chg = total_change / n
    print(f"Перша ціна: {prices[0]:.2f} USD")
    print(f"Остання ціна: {prices[-1]:.2f} USD")
    print(f"Загальна зміна: {total_change:+.2f} USD ({pct_change:+.2f}%)")
    slope = MNK_AV_Detect(prices)


def stat_characteristics(prices: np.ndarray, label: str) -> dict:
    Yout = MNK_Stat_characteristics(prices)
    residuals = np.array([prices[i] - Yout[i, 0] for i in range(len(prices))])

    m_mean = float(np.mean(residuals))
    m_median = float(np.median(residuals))
    dS = float(np.var(residuals))
    scvS = mt.sqrt(dS)
    skew = float(pd.Series(residuals).skew())
    kurt = float(pd.Series(residuals).kurt())

    print("=" * 60)
    print(f"4. Статистичні характеристики: {label}")
    print(f"Кількість елементів вибірки: {len(prices)}")
    print(f"Математичне сподівання (mean): {m_mean:.4f}")
    print(f"Медіана залишків: {m_median:.4f}")
    print(f"Дисперсія: {dS:.4f}")
    print(f"СКВ: {scvS:.4f}")
    print(f"Асиметрія: {skew:.4f}")
    print(f"Ексцес: {kurt:.4f}")
    print(f"Мін / Макс ціни: {prices.min():.2f} / {prices.max():.2f} USD")
    print(f"Розмах: {prices.max() - prices.min():.2f} USD")

    # гістограма залишків
    plt.figure(figsize=(7, 4))
    plt.hist(residuals, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
    plt.axvline(m_mean, color="crimson", linestyle="--", label=f"mean={m_mean:.2f}")
    plt.title(f"Гістограма залишків: {label}", fontsize=12)
    plt.xlabel("Відхилення від МНК-тренду, USD")
    plt.ylabel("Частота")
    plt.legend()
    plt.tight_layout()
    hist_file = "histogram_residuals.png"
    plt.savefig(hist_file, dpi=120)
    plt.close()
    print(f"Гістограму збережено: {os.path.abspath(hist_file)}")

    return {"mean": m_mean, "median": m_median, "var": dS, "std": scvS,
            "skew": skew, "kurt": kurt, "min": prices.min(), "max": prices.max()}


def MNK_Stat_characteristics(S0: np.ndarray) -> np.ndarray:
    n = len(S0)
    Yin = np.zeros((n, 1))
    F = np.ones((n, 3))
    for i in range(n):
        Yin[i, 0] = float(S0[i])
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    C = np.linalg.inv(FT.dot(F)).dot(FT).dot(Yin)
    Yout = F.dot(C)
    return Yout


def MNK(S0: np.ndarray) -> np.ndarray:
    n = len(S0)
    Yin = np.zeros((n, 1))
    F = np.ones((n, 3))
    for i in range(n):
        Yin[i, 0] = float(S0[i])
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    C = np.linalg.inv(FT.dot(F)).dot(FT).dot(Yin)
    Yout = F.dot(C)
    print(f"Регресійна МНК-модель:")
    print(f"y(t) = {C[0,0]:.6f}  +  {C[1,0]:.6f} * t  +  {C[2,0]:.8f} * t^2")
    return Yout


def MNK_AV_Detect(S0: np.ndarray) -> float:
    n = len(S0)
    Yin = np.zeros((n, 1))
    F = np.ones((n, 3))
    for i in range(n):
        Yin[i, 0] = float(S0[i])
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    C = np.linalg.inv(FT.dot(F)).dot(FT).dot(Yin)
    return float(C[1, 0])


def MNK_Extrapol(S0: np.ndarray, koef: int) -> np.ndarray:
    n = len(S0)
    Yin = np.zeros((n, 1))
    F = np.ones((n, 3))
    for i in range(n):
        Yin[i, 0] = float(S0[i])
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    C = np.linalg.inv(FT.dot(F)).dot(FT).dot(Yin)
    Yout = np.zeros((n + koef, 1))
    for i in range(n + koef):
        Yout[i, 0] = C[0,0] + C[1,0]*i + C[2,0]*i*i
    print(f"МНК-прогноз: y(t) = {C[0,0]:.4f} + {C[1,0]:.6f}*t + {C[2,0]:.8f}*t^2")
    return Yout


def r2_score_mnk(prices: np.ndarray, Yout: np.ndarray, label: str) -> float:
    n = len(prices)
    numerator = sum((prices[i] - Yout[i, 0])**2 for i in range(n))
    mean_y = np.mean(prices)
    denominator = sum((prices[i] - mean_y)**2 for i in range(n))
    R2 = 1 - numerator / denominator
    print(f"[{label}] Коефіцієнт детермінації R^2 = {R2:.6f}")
    return R2


def sliding_window_clean(S0: np.ndarray, n_wind: int) -> np.ndarray:
    n = len(S0)
    j_wind = mt.ceil(n - n_wind) + 1
    Midi = np.zeros(n)
    wind = np.zeros(n_wind)
    for j in range(j_wind):
        for i in range(n_wind):
            wind[i] = S0[j + i]
        Midi[j + n_wind - 1] = np.median(wind)
    S_clean = np.copy(S0)
    for j in range(n_wind, n):
        S_clean[j] = Midi[j]
    return S_clean


def synthesize_and_verify(prices: np.ndarray, stats: dict) -> None:
    print("=" * 60)
    print("5. Синтез та верифікація моделі даних")

    n = len(prices)
    n_wind = 5
    koef = mt.ceil(n * 0.2) # прогноз на 20% від обсягу вибірки

    prices_clean = sliding_window_clean(prices, n_wind)

    print("\n   --- МНК-згладжування ---")
    Yout_smooth = MNK(prices_clean)
    R2_smooth = r2_score_mnk(prices_clean, Yout_smooth, "МНК-згладжування")

    print("\n   --- МНК-прогнозування ---")
    Yout_extrapol = MNK_Extrapol(prices_clean, koef)
    R2_extrapol = r2_score_mnk(prices_clean, Yout_extrapol[:n], "МНК-прогнозування")

    # --- синтетичні дані: тренд МНК + нормальний шум зі стат. хар-ками реальних залишків ---
    synthetic = np.array([Yout_smooth[i, 0] + np.random.normal(stats["mean"], stats["std"])
                          for i in range(n)])

    # --- верифікація: порівняння стат. характеристик синтетичних і реальних залишків ---
    print("\n   --- Верифікація: порівняння стат. характеристик ---")
    Yout_synth = MNK(synthetic)
    synth_resid = np.array([synthetic[i] - Yout_synth[i, 0] for i in range(n)])
    print(f"Реальні дані: mean={stats['mean']:.4f}, σ={stats['std']:.4f}")
    print(f"Синт. модель: mean={np.mean(synth_resid):.4f}, σ={np.std(synth_resid):.4f}")
    R2_synth = r2_score_mnk(synthetic, Yout_synth, "синтетична МНК-модель")

    #графік МНК-тренду та прогнозу
    plt.figure(figsize=(13, 5))
    x_hist = np.arange(n)
    x_full = np.arange(n + koef)
    plt.plot(x_hist, prices_clean, color="steelblue", linewidth=1, label="Реальні дані (очищені)")
    plt.plot(x_hist, Yout_smooth[:, 0], color="crimson", linewidth=2, linestyle="--", label="МНК-тренд")
    plt.plot(x_full[n:], Yout_extrapol[n:, 0], color="orange", linewidth=2, linestyle=":", label=f"МНК-прогноз (+{koef} точок)")
    plt.axvline(n - 1, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    plt.title("МНК: тренд та прогноз цін ETF GLD", fontsize=13)
    plt.xlabel("Індекс спостереження")
    plt.ylabel("Ціна закриття, USD")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mnk_trend_forecast.png", dpi=120)
    plt.close()
    print(f"Графік збережено: {os.path.abspath('mnk_trend_forecast.png')}")

    #порівняльний графік
    plt.figure(figsize=(13, 5))
    plt.plot(prices_clean, color="steelblue", linewidth=1, alpha=0.8, label="Реальні дані GLD")
    plt.plot(synthetic, color="green", linewidth=1, alpha=0.6, linestyle="--", label="Синтетична модель")
    plt.plot(Yout_smooth[:, 0], color="crimson", linewidth=2, label="МНК-тренд")
    plt.title("Верифікація: реальні дані vs синтетична модель", fontsize=13)
    plt.xlabel("Індекс спостереження")
    plt.ylabel("Ціна, USD")
    plt.legend()
    plt.tight_layout()
    plt.savefig("verification_real_vs_synthetic.png", dpi=120)
    plt.close()
    print(f"Графік збережено: {os.path.abspath('verification_real_vs_synthetic.png')}")


if __name__ == "__main__":

    df_raw = parse_gold_prices()
    df_clean = clean_dataframe(df_raw)

    save_to_csv(df_clean, "gold_prices.csv")

    prices = df_clean["Close"].values.astype(float)

    analyze_trend(prices)
    Yout_trend = MNK_Stat_characteristics(prices)
    plot_trend(prices, Yout_trend, "Динаміка цін ETF GLD та МНК-тренд", "trend_analysis.png")

    stats = stat_characteristics(prices, "ETF GLD (ціна закриття)")

    synthesize_and_verify(prices, stats)