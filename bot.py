import os
import time
import telebot
import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.linear_model import LinearRegression
from datetime import datetime, timezone, timedelta
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ======================
# CONFIG
# ======================
API_TOKEN = os.getenv("BOT_TOKEN")  # Railway -> Variables -> BOT_TOKEN
if not API_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set. Add it in Railway -> Variables.")

bot = telebot.TeleBot(API_TOKEN)

ALLOWED_TICKERS = {"META", "SNAP", "PINS"}

HISTORY_PERIOD = "6mo"
INTERVAL = "1d"

PREDICT_DAYS_7D = 7
NEWS_LOOKBACK_DAYS = 7
NEWS_LIMIT = 12

# Cache to reduce Yahoo calls / avoid 429
CACHE_TTL_SECONDS = 60
_cache = {}  # key -> (ts, value)

analyzer = SentimentIntensityAnalyzer()


# ======================
# HELPERS
# ======================
def cache_get(key: str):
    item = _cache.get(key)
    if not item:
        return None
    ts, val = item
    if time.time() - ts > CACHE_TTL_SECONDS:
        _cache.pop(key, None)
        return None
    return val

def cache_set(key: str, val):
    _cache[key] = (time.time(), val)

def sanitize_md(text: str) -> str:
    # Prevent breaking Markdown
    if not text:
        return ""
    return (text.replace("*", "")
                .replace("_", "")
                .replace("`", "")
                .replace("[", "(")
                .replace("]", ")"))

def parse_predict_command(text: str):
    """
    Supports:
      /predict META 1d
      /predict META 7d
    Returns: (ticker, mode) where mode is "1d" or "7d"
    """
    parts = text.split()
    if len(parts) != 3:
        return None, None

    cmd, ticker, mode = parts[0], parts[1], parts[2]
    ticker = ticker.upper().strip()
    mode = mode.lower().strip()

    if mode not in {"1d", "7d"}:
        return ticker, None

    return ticker, mode


# ======================
# INLINE MENU
# ======================
def main_menu():
    keyboard = InlineKeyboardMarkup(row_width=2)
    keyboard.add(
        InlineKeyboardButton("📊 Predict META (1D)", callback_data="predict1_META"),
        InlineKeyboardButton("📅 Predict META (7D)", callback_data="predict7_META"),
        InlineKeyboardButton("📊 Predict SNAP (1D)", callback_data="predict1_SNAP"),
        InlineKeyboardButton("📅 Predict SNAP (7D)", callback_data="predict7_SNAP"),
        InlineKeyboardButton("📊 Predict PINS (1D)", callback_data="predict1_PINS"),
        InlineKeyboardButton("📅 Predict PINS (7D)", callback_data="predict7_PINS"),
        InlineKeyboardButton("ℹ️ Status", callback_data="status"),
    )
    return keyboard


# ======================
# DATA FROM YAHOO FINANCE
# ======================
def fetch_price_df(ticker: str) -> pd.DataFrame:
    cache_key = f"prices:{ticker}:{HISTORY_PERIOD}:{INTERVAL}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    df = yf.download(
        ticker,
        period=HISTORY_PERIOD,
        interval=INTERVAL,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df is None or df.empty:
        raise ValueError("Yahoo Finance returned empty data (try later).")
    if "Close" not in df.columns:
        raise ValueError("Yahoo Finance data has no 'Close' column.")

    df = df.dropna(subset=["Close"]).copy()
    if len(df) < 20:
        raise ValueError("Not enough data (need at least 20 days).")

    cache_set(cache_key, df)
    return df


# ======================
# NEWS FROM YAHOO (via yfinance)
# ======================
def fetch_news_yahoo(ticker: str) -> list[dict]:
    cache_key = f"news:{ticker}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    t = yf.Ticker(ticker)
    try:
        news = t.news
    except Exception:
        news = t.get_news()

    if not news:
        news = []

    cache_set(cache_key, news)
    return news


def score_news_titles(news: list[dict], lookback_days: int = 7, limit: int = 10):
    cutoff = int(time.time()) - lookback_days * 24 * 3600

    items = []
    for n in news:
        ts = n.get("providerPublishTime") or 0
        if ts < cutoff:
            continue

        title = n.get("title") or ""
        link = n.get("link") or n.get("url") or ""
        publisher = n.get("publisher") or ""
        if not title:
            continue

        s = analyzer.polarity_scores(title)["compound"]
        items.append({"title": title, "link": link, "publisher": publisher, "score": s, "ts": ts})

    items.sort(key=lambda x: x["ts"], reverse=True)
    items = items[:limit]

    if not items:
        return 0.0, [], [], 0

    avg = float(np.mean([x["score"] for x in items]))
    top_pos = sorted(items, key=lambda x: x["score"], reverse=True)[:3]
    top_neg = sorted(items, key=lambda x: x["score"])[:3]
    return avg, top_pos, top_neg, len(items)


# ======================
# MODEL (Linear Regression) + PREDICT
# ======================
def predict_prices(ticker: str, days: int):
    df = fetch_price_df(ticker)
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()

    y = close.values.reshape(-1, 1)
    X = np.arange(len(y)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    r2 = float(model.score(X, y))
    slope = float(model.coef_[0][0])

    start = len(y)
    future_X = np.arange(start, start + days).reshape(-1, 1)
    preds = model.predict(future_X).reshape(-1).astype(float).tolist()

    last_price = float(y[-1][0])

    last_date = close.index[-1]
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days).to_list()

    return preds, last_price, r2, slope, future_dates


# ======================
# TEXT FORMATTERS
# ======================
def build_explanation(ticker: str, slope: float, news_avg: float, pos, neg, used_count: int) -> str:
    if slope > 0:
        trend_txt = "Trend: *Uptrend* (slope > 0)"
        trend_bias = 1
    elif slope < 0:
        trend_txt = "Trend: *Downtrend* (slope < 0)"
        trend_bias = -1
    else:
        trend_txt = "Trend: *Flat* (slope ~ 0)"
        trend_bias = 0

    if used_count == 0:
        news_txt = "News: *No recent news found* (or Yahoo rate limit)."
        news_bias = 0
    else:
        if news_avg > 0.10:
            news_txt = f"News sentiment: *Positive* (avg `{news_avg:+.2f}` from {used_count} headlines)"
            news_bias = 1
        elif news_avg < -0.10:
            news_txt = f"News sentiment: *Negative* (avg `{news_avg:+.2f}` from {used_count} headlines)"
            news_bias = -1
        else:
            news_txt = f"News sentiment: *Mixed/Neutral* (avg `{news_avg:+.2f}` from {used_count} headlines)"
            news_bias = 0

    combined = trend_bias + news_bias
    if combined >= 2:
        final = "Reasoning: uptrend + positive news → *higher probability of growth*."
    elif combined <= -2:
        final = "Reasoning: downtrend + negative news → *higher probability of decline*."
    elif combined == 1:
        final = "Reasoning: slightly positive signals → *mild bullish bias*."
    elif combined == -1:
        final = "Reasoning: slightly negative signals → *mild bearish bias*."
    else:
        final = "Reasoning: weak/conflicting signals → *uncertain / sideways risk*."

    def format_news_list(label, arr):
        if not arr:
            return ""
        out = [f"*{label}:*"]
        for x in arr:
            title = sanitize_md(x["title"])
            publisher = sanitize_md(x.get("publisher", ""))
            score = x["score"]
            link = x.get("link", "")
            out.append(f"• `{score:+.2f}` — {title} ({publisher})\n  {link}")
        return "\n".join(out)

    parts = [
        "📊 *Why it may move?*",
        f"• {trend_txt}",
        f"• {news_txt}",
        f"• {final}",
    ]

    if used_count > 0:
        parts.append("")
        parts.append(format_news_list("Top positive headlines", pos))
        parts.append("")
        parts.append(format_news_list("Top negative headlines", neg))

    return "\n".join([p for p in parts if p])


def build_predict_1d_text(ticker: str, predicted: float, last_price: float, r2: float, explanation: str) -> str:
    direction = "📈 UP" if predicted > last_price else "📉 DOWN"
    kz_time = datetime.now(timezone.utc) + timedelta(hours=5)

    return (
        f"*{ticker} Predict (1D)*\n\n"
        f"Last close: `{last_price:.2f}`\n"
        f"Predicted next close: `{predicted:.2f}`\n"
        f"Direction: {direction}\n"
        f"Confidence (R²): `{r2*100:.1f}%`\n\n"
        f"{explanation}\n\n"
        f"⏱ {kz_time.strftime('%Y-%m-%d %H:%M')}"
    )


def build_predict_7d_text(ticker: str, preds: list[float], last_price: float, r2: float, future_dates: list, explanation: str) -> str:
    day7 = preds[-1]
    direction = "📈 UP" if day7 > last_price else "📉 DOWN"
    kz_time = datetime.now(timezone.utc) + timedelta(hours=5)

    lines = [f"• {d.strftime('%Y-%m-%d')}: `{p:.2f}`" for d, p in zip(future_dates, preds)]
    predict_block = "\n".join(lines)

    return (
        f"*{ticker} Predict (7D)*\n\n"
        f"Last close: `{last_price:.2f}`\n"
        f"Day 7 prediction: `{day7:.2f}`\n"
        f"Direction (vs last): {direction}\n"
        f"Confidence (R²): `{r2*100:.1f}%`\n\n"
        f"*Next 7 predicted closes:*\n{predict_block}\n\n"
        f"{explanation}\n\n"
        f"⏱ {kz_time.strftime('%Y-%m-%d %H:%M')}"
    )


# ======================
# CORE ACTIONS
# ======================
def do_predict(ticker: str, mode: str) -> str:
    """
    mode: "1d" or "7d"
    """
    days = 1 if mode == "1d" else PREDICT_DAYS_7D

    preds, last_price, r2, slope, future_dates = predict_prices(ticker, days=days)

    news = fetch_news_yahoo(ticker)
    news_avg, top_pos, top_neg, used_count = score_news_titles(
        news, lookback_days=NEWS_LOOKBACK_DAYS, limit=NEWS_LIMIT
    )
    explanation = build_explanation(ticker, slope, news_avg, top_pos, top_neg, used_count)

    if mode == "1d":
        return build_predict_1d_text(ticker, preds[0], last_price, r2, explanation)
    else:
        return build_predict_7d_text(ticker, preds, last_price, r2, future_dates, explanation)


def predict_usage_text():
    return (
        "✅ Формат команд:\n\n"
        "• `/predict META 1d`\n"
        "• `/predict META 7d`\n\n"
        "Доступные тикеры: " + ", ".join(sorted(ALLOWED_TICKERS))
    )


# ======================
# COMMANDS
# ======================
@bot.message_handler(commands=["start"])
def start(message):
    bot.send_message(
        message.chat.id,
        "👋 *Hello!*\n\n"
        "You are using *Predict AI* — a Telegram bot that predicts short-term stock moves.\n\n"
        "📌 *Commands:*\n"
        "• `/predict META 1d`  — Predict next close\n"
        "• `/predict META 7d`  — Predict next 7 trading closes\n\n"
        "📌 *What it uses:*\n"
        "• Yahoo Finance prices\n"
        "• Yahoo headlines (via yfinance)\n"
        "• Trend + news sentiment explanation\n\n"
        "⚠️ *Not financial advice.*\n\n"
        "Choose an action below 👇",
        reply_markup=main_menu(),
        parse_mode="Markdown"
    )


@bot.message_handler(commands=["status"])
def status(message):
    bot.reply_to(
        message,
        f"✅ Bot RUNNING\nSource: Yahoo Finance\nTickers: {', '.join(sorted(ALLOWED_TICKERS))}\n"
        f"Command: /predict <TICKER> <1d|7d>"
    )


@bot.message_handler(commands=["help"])
def help_cmd(message):
    bot.send_message(message.chat.id, predict_usage_text(), parse_mode="Markdown")


@bot.message_handler(commands=["predict"])
def predict_command(message):
    ticker, mode = parse_predict_command(message.text)

    if ticker is None and mode is None:
        bot.send_message(message.chat.id, predict_usage_text(), parse_mode="Markdown")
        return

    if ticker not in ALLOWED_TICKERS:
        bot.send_message(message.chat.id, f"Разрешены: {', '.join(sorted(ALLOWED_TICKERS))}")
        return

    if mode not in {"1d", "7d"}:
        bot.send_message(message.chat.id, "Укажи режим `1d` или `7d`.\n\n" + predict_usage_text(), parse_mode="Markdown")
        return

    try:
        text = do_predict(ticker, mode)
        bot.send_message(message.chat.id, text, parse_mode="Markdown")
    except Exception as e:
        bot.send_message(message.chat.id, f"Ошибка: {e}")


# ======================
# BUTTON HANDLER
# ======================
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    try:
        if call.data == "status":
            bot.send_message(
                call.message.chat.id,
                f"✅ Bot RUNNING\nSource: Yahoo Finance\nTickers: {', '.join(sorted(ALLOWED_TICKERS))}\n"
                f"Command: /predict <TICKER> <1d|7d>"
            )
            return

        if call.data.startswith("predict1_"):
            ticker = call.data.split("_", 1)[1].upper()
            if ticker not in ALLOWED_TICKERS:
                bot.send_message(call.message.chat.id, "Ticker not allowed.")
                return
            text = do_predict(ticker, "1d")
            bot.send_message(call.message.chat.id, text, parse_mode="Markdown")
            return

        if call.data.startswith("predict7_"):
            ticker = call.data.split("_", 1)[1].upper()
            if ticker not in ALLOWED_TICKERS:
                bot.send_message(call.message.chat.id, "Ticker not allowed.")
                return
            text = do_predict(ticker, "7d")
            bot.send_message(call.message.chat.id, text, parse_mode="Markdown")
            return

    except Exception as e:
        bot.send_message(call.message.chat.id, f"Ошибка: {e}")


# ======================
# RUN
# ======================
if __name__ == "__main__":
    print("Bot started ✅ (Yahoo prices + Yahoo news enabled)")
    bot.infinity_polling()
