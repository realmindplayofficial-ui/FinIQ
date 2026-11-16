import yfinance as yf
import pandas as pd
import numpy as np
import json
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

# NSE Top Equity & F&O Stocks List
FNO_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "SBIN.NS", "BAJFINANCE.NS", "TATAMOTORS.NS", "AXISBANK.NS", "LT.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "ADANIENT.NS", "TATASTEEL.NS", "M&M.NS",
    "BHARTIARTL.NS", "TITAN.NS", "NTPC.NS", "ONGC.NS", "JSWSTEEL.NS"
]

def fetch_rss_news(url, limit=6):
    headlines = []
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            for item in root.findall('.//item')[:limit]:
                title = item.find('title').text
                clean_title = title.split(' - ') if ' - ' in title else title
                headlines.append(clean_title)
    except Exception:
        pass
    return headlines

def run_analysis():
    results = []
    fno_setups = []
    buy_count = 0
    sell_count = 0

    # Fetch Nifty 50 Index Data
    nifty = yf.Ticker("^NSEI").history(period="5d")
    nifty_price = round(nifty['Close'].iloc[-1], 0)
    nifty_change = round(((nifty['Close'].iloc[-1] - nifty['Close'].iloc[-2]) / nifty['Close'].iloc[-2]) * 100, 2)

    for symbol in FNO_TICKERS:
        try:
            df = yf.Ticker(symbol).history(period="1y")
            if len(df) < 200:
                continue
            
            # Moving Averages & Volume
            df['SMA50'] = df['Close'].rolling(50).mean()
            df['SMA200'] = df['Close'].rolling(200).mean()
            df['Vol20'] = df['Volume'].rolling(20).mean()
            
            # RSI (14)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            # ATR (14)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            tr = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1)
            df['ATR'] = tr.rolling(14).mean()

            last, prev = df.iloc[-1], df.iloc[-2]
            close = float(last['Close'])
            change_pct = ((close - float(prev['Close'])) / float(prev['Close'])) * 100
            vol_ratio = round(float(last['Volume']) / float(last['Vol20']), 2)

            # 1. Equity Score (-4 to +4)
            score = (1 if close > last['SMA50'] else -1) + (1 if last['SMA50'] > last['SMA200'] else -1)
            if 45 <= last['RSI'] <= 68:
                score += 1
            elif last['RSI'] > 75:
                score -= 1
            if vol_ratio >= 1.5:
                score += 1 if change_pct >= 0 else -1

            signal = "BUY" if score >= 3 else ("SELL" if score <= -2 else "HOLD")
            if signal == "BUY":
                buy_count += 1
            elif signal == "SELL":
                sell_count += 1

            atr = float(last['ATR'])
            stoploss = min(max(close - 1.5 * atr, float(last['SMA50']) * 0.97), close - 0.75 * atr)
            risk = close - stoploss

            results.append({
                "symbol": symbol.replace(".NS", ""),
                "price": round(close, 2),
                "change_pct": round(change_pct, 2),
                "score": score,
                "signal": signal,
                "stoploss": round(stoploss, 2),
                "sl_pct": round(-((close - stoploss) / close) * 100, 1),
                "target1": round(close + 2 * risk, 2),
                "t1_pct": round(((2 * risk) / close) * 100, 1),
                "target2": round(close + 3 * risk, 2),
                "t2_pct": round(((3 * risk) / close) * 100, 1),
                "risk_per_share": round(risk, 2)
            })

            # 2. Stock F&O IMP Points Calculation (0 to 10 Points)
            imp_points = abs(score) * 2
            if vol_ratio >= 1.5:
                imp_points += 2
            elif vol_ratio >= 1.2:
                imp_points += 1
            if abs(change_pct) >= 1.5:
                imp_points += 1

            is_bullish = (score >= 2 and close > last['SMA50'])
            is_bearish = (score <= -2 and close < last['SMA50'])

            if is_bullish or is_bearish:
                strike_step = 50 if close > 2000 else (20 if close > 1000 else 10)
                atm_strike = round(close / strike_step) * strike_step
                opt_type = "CE" if is_bullish else "PE"

                # Spot Stoploss & Targets
                if is_bullish:
                    spot_sl = round(close - (1.2 * atr), 2)
                    spot_risk = close - spot_sl
                    spot_t1 = round(close + (1.8 * spot_risk), 2)
                    spot_t2 = round(close + (2.8 * spot_risk), 2)
                else:
                    spot_sl = round(close + (1.2 * atr), 2)
                    spot_risk = spot_sl - close
                    spot_t1 = round(close - (1.8 * spot_risk), 2)
                    spot_t2 = round(close - (2.8 * spot_risk), 2)

                # Estimated ATM Option Premium & Targets (Delta ~0.50)
                est_opt_entry = round(close * 0.022, 1)
                opt_risk = round(spot_risk * 0.50, 1)
                est_opt_sl = max(1.0, round(est_opt_entry - opt_risk, 1))
                est_opt_t1 = round(est_opt_entry + (1.8 * opt_risk), 1)
                est_opt_t2 = round(est_opt_entry + (2.8 * opt_risk), 1)

                fno_setups.append({
                    "symbol": symbol.replace(".NS", ""),
                    "strike_name": f"{symbol.replace('.NS', '')} {atm_strike} {opt_type}",
                    "opt_type": opt_type,
                    "spot": round(close, 2),
                    "change_pct": round(change_pct, 2),
                    "imp_points": imp_points,
                    "vol_spike": f"{vol_ratio}x",
                    "rsi": round(float(last['RSI']), 1),
                    "spot_sl": spot_sl,
                    "spot_t1": spot_t1,
                    "spot_t2": spot_t2,
                    "opt_entry": est_opt_entry,
                    "opt_sl": est_opt_sl,
                    "opt_t1": est_opt_t1,
                    "opt_t2": est_opt_t2
                })
        except Exception as e:
            print(f"Skipping {symbol}: {e}")

    # Sort Equity by Score & F&O by Highest IMP Points
    results.sort(key=lambda x: x['score'], reverse=True)
    fno_setups.sort(key=lambda x: (x['imp_points'], float(x['vol_spike'].replace('x', ''))), reverse=True)

    # 3. Fetch Live Global & National News Headlines
    global_news = fetch_rss_news("https://google.com", 6)
    national_news = fetch_rss_news("https://google.com", 6)

    if not global_news:
        global_news = [
            "US Treasury yields steady ahead of key inflation data.",
            "Global crude oil prices consolidate amid supply outlook."
        ]
    if not national_news:
        national_news = [
            "Nifty & Sensex eye domestic institutional inflows.",
            "RBI maintains focus on liquidity and inflation stability."
        ]

    # Save Final Output for FinIQ Dashboard
    with open("data.json", "w") as f:
        json.dump({
            "app_name": "FinIQ",
            "updated_at": datetime.now().strftime("%a %d %b · %H:%M:%S IST"),
            "nifty": int(nifty_price),
            "day_change": nifty_change,
            "total_stocks": len(results),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "best_pick": results if results else None,
            "all_stocks": results,
            "best_fno": fno_setups if fno_setups else None,
            "all_fno": fno_setups,
            "global_news": global_news,
            "national_news": national_news
        }, f, indent=2)
    print("FinIQ data.json ??????? ???? ?????!")

if __name__ == "__main__":
    run_analysis()