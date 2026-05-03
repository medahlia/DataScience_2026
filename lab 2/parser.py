import os
import requests
import pandas as pd


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