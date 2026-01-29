# bot.py
import os
import time
import telebot
import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.linear_model import LinearRegression
from datetime import datetime, timezone, timedelta
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton


# ======================
# CONFIG
# ======================
API_TOKEN = os.getenv("BOT_TOKEN")  # Railway/Render -> Variables -> BOT_TOKEN
if not API_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set. Add it in Variables (BOT_TOKEN).")

bot = telebot.TeleBot(API_TOKEN)

ALLOWED_TICKERS = {"META", "SNAP", "PINS"}

HISTORY_PERIOD = "6mo"
INTERVAL = "1d"

NEWS_LOOKBACK_DAYS = 7
NEWS_LIMIT = 12

# cache (чтобы меньше дергать Yahoo)
CACHE_TTL_SECONDS = 60
_cache = {}  # key -> (ts, value)


# ======================
# NEWS RULES (простая "логика понимания" новостей)
# ======================
NEWS_RULES = [
    # Very positive
    {
        "keywords": [
            "beats earnings", "earnings beat", "revenue beat",
            "raised guidance", "guidance raised",
            "upgrade", "price target raised"
        ],
        "score": +3,
        "tag": "Strong positive"
    },
    {
        "keywords": ["partnership", "deal", "contract", "acquisition", "buyback", "share repurchase"],
        "score": +2,
        "tag": "Positive catalyst"
    },
    {
        "keywords": ["launch", "released", "new product", "record revenue", "strong demand"],
        "score": +2,
        "tag": "Growth signal"
    },

    # Neutral
    {"keywords": ["expects", "plans", "considering", "announced", "update"], "score": 0, "tag": "Neutral"},

    # Negative
    {
        "keywords": [
            "missed earnings", "earnings miss", "revenue miss",
            "cut guidance", "guidance cut",
            "downgrade", "price target cut"
        ],
        "score": -3,
        "tag": "Strong negative"
    },
    {"keywords": ["lawsuit", "probe", "investigation", "regulators", "fine", "ban", "antitrust"], "score": -2,
     "tag": "Legal/regulatory risk"},
    {"keywords": ["weak demand", "slowdown", "warning", "decline", "fell", "drops"], "score": -2,
     "tag": "Weakness signal"},
    {"keywords": ["layoffs", "cuts jobs", "cost cutting"], "score": -1, "tag": "Cost-cutting (mixed)"},
]


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
    # чтобы не ломать Telegram Markdown
    if not text:
        return ""
    return (text.replace("*", "")
                .replace("_", "")
                .replace("`", "")
                .replace("[", "(")
                .replace("]", ")"))


def parse_predict_command(text: str):
    """
    /predict META 1d
    /predict META 7d
    returns: (ticker, mode) mode in {"1d","7d"} or (None,None)
    """
    parts = text.split()
    if len(parts) != 3:
        return None, None
    ticker = parts[1].upper().strip()
    mode = parts[2].lower().strip()
    if mode not in {"1d", "7d"}:
        return ticker, None
    return ticker, mode


# ======================
# INLINE MENU
# ======================
def main_menu():
    keyboard = InlineKeyboardMarkup(row_width=2)
    keyboard.add(
        InlineKeyboardButton("📊 Predict META (1D)", callback_data="predict_META_1d"),
        InlineKeyboardButton("📅 Predict META (7D)", callback_data="predict_META_7d"),
        InlineKeyboardButton("📊 Predict SNAP (1D)", callback_data="predict_SNAP_1d"),
        InlineKeyboardButton("📅 Predict SNAP (7D)", callback_data="predict_SNAP_7d"),
        InlineKeyboardButton("📊 Predict PINS (1D)", callback_data="predict_PINS_1d"),
        InlineKeyboardButton("📅 Predict PINS (7D)", callback_data="predict_PINS_7d"),
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

    df = yf.download(ticker, period=HISTORY_PERIOD, interval=INTERVAL, progress=False)

    # иногда yfinance отдаёт MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df is None or df.empty:
        raise ValueError("Yahoo Finance returned empty data (try later).")

    if "Close" not in df.columns:
        raise ValueError("Yahoo Finance data has no 'Close' column.")

    df = df.dropna(subset=["Close"]).copy()
    if len(df) < 20:
        raise ValueError("Not enough data from Yahoo Finance (need at least 20 days).")

    cache_set(cache_key, df)
    return df


# ======================
# NEWS FROM YAHOO (via yfinance)
# ======================
def fetch_news_yahoo(ticker: str) -> list[dict]:
    """
    yfinance берет новости с Yahoo Finance и отдаёт:
    title / publisher / link / providerPublishTime ...
    """
    cache_key = f"news:{ticker}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    t = yf.Ticker(ticker)
    try:
        news = t.news  # чаще работает
    except Exception:
        # на некоторых версиях
        try:
            news = t.get_news()
        except Exception:
            news = []

    if not news:
        news = []

    cache_set(cache_key, news)
    return news


def analyze_news_semantic(news: list[dict], lookback_days: int = 7, limit: int = 12):
    """
    "Изучение" новостей через правила:
    - ищем ключевые слова и даем score новости
    Возвращает:
      avg_score (средняя оценка),
      top_items (самые "влиятельные" по |score|),
      used_count
    """
    cutoff = int(time.time()) - lookback_days * 24 * 3600

    items = []
    for n in news:
        ts = n.get("providerPublishTime") or 0
        if ts < cutoff:
            continue

        raw_title = n.get("title") or ""
        title = raw_title.lower().strip()
        if not title:
            continue

        link = n.get("link") or n.get("url") or ""
        publisher = n.get("publisher") or ""

        score = 0
        tags = []

        for rule in NEWS_RULES:
            for kw in rule["keywords"]:
                if kw in title:
                    score += rule["score"]
                    tags.append(rule["tag"])
                    break

        items.append({
            "title": raw_title,
            "publisher": publisher,
            "link": link,
            "ts": ts,
            "score": int(score),
            "tags": list(dict.fromkeys(tags)),
        })

    # свежие сверху, режем limit
    items.sort(key=lambda x: x["ts"], reverse=True)
    items = items[:limit]

    if not items:
        return 0.0, [], 0

    avg_score = float(np.mean([x["score"] for x in items]))
    top_items = sorted(items, key=lambda x: abs(x["score"]), reverse=True)[:5]
    return avg_score, top_items, len(items)


def build_news_reasoning(avg_score: float, top_items: list[dict], used_count: int):
    """
    Возвращает (текст_новостей, news_bias)
    news_bias: -1 / 0 / +1
    """
    if used_count == 0:
        return "News analysis: *No recent news found (or Yahoo limit).*", 0

    if avg_score >= 1.0:
        bias = 1
        head = f"News analysis: *Bullish* (avg `{avg_score:+.2f}` from {used_count} headlines)"
    elif avg_score <= -1.0:
        bias = -1
        head = f"News analysis: *Bearish* (avg `{avg_score:+.2f}` from {used_count} headlines)"
    else:
        bias = 0
        head = f"News analysis: *Mixed/Neutral* (avg `{avg_score:+.2f}` from {used_count} headlines)"

    lines = [head, "", "*Most impactful headlines:*"]
    for it in top_items:
        title = sanitize_md(it["title"])
        tags = ", ".join(it["tags"]) if it["tags"] else "Unclassified"
        publisher = sanitize_md(it.get("publisher", ""))
        link = it.get("link", "")

        lines.append(
            f"• `{it['score']:+d}` — {title} ({publisher})\n"
            f"  Tags: {sanitize_md(tags)}\n"
            f"  {link}"
        )

    return "\n".join(lines), bias


# ======================
# MODEL (Linear Regression) + PREDICT N DAYS
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

    # торговые дни (business days)
    last_date = close.index[-1]
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days).to_list()

    return preds, last_price, r2, slope, future_dates


# ======================
# TEXT FORMATTERS
# ======================
def build_predict_text(ticker: str, mode: str, preds: list[float], last_price: float, r2: float,
                      future_dates: list, explanation: str) -> str:
    kz_time = datetime.now(timezone.utc) + timedelta(hours=5)

    if mode == "1d":
        predicted = preds[0]
        direction = "📈 UP" if predicted > last_price else "📉 DOWN"
        return (
            f"*{ticker} Predict (1D)*\n\n"
            f"Last close: `{last_price:.2f}`\n"
            f"Predicted next close: `{predicted:.2f}`\n"
            f"Direction: {direction}\n"
            f"Confidence (R²): `{r2*100:.1f}%`\n\n"
            f"{explanation}\n\n"
            f"⏱ {kz_time.strftime('%Y-%m-%d %H:%M')}"
        )

    # 7d
    day7 = preds[-1]
    direction = "📈 UP" if day7 > last_price else "📉 DOWN"
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


def build_combined_explanation(slope: float, news_block: str, news_bias: int) -> str:
    # тренд
    if slope > 0:
        trend_bias = 1
        trend_txt = f"Trend: *Uptrend* (slope `{slope:+.4f}`)"
    elif slope < 0:
        trend_bias = -1
        trend_txt = f"Trend: *Downtrend* (slope `{slope:+.4f}`)"
    else:
        trend_bias = 0
        trend_txt = f"Trend: *Flat* (slope `{slope:+.4f}`)"

    combined = trend_bias + news_bias

    if combined >= 2:
        final = "Overall: uptrend + bullish news → *higher probability of growth*."
    elif combined <= -2:
        final = "Overall: downtrend + bearish news → *higher probability of decline*."
    elif combined == 1:
        final = "Overall: signals slightly positive → *mild bullish bias*."
    elif combined == -1:
        final = "Overall: signals slightly negative → *mild bearish bias*."
    else:
        final = "Overall: weak/conflicting signals → *uncertain / sideways risk*."

    return (
        "📊 *Why it may move?*\n"
        f"• {trend_txt}\n"
        f"• {final}\n\n"
        f"{news_block}"
    )


def usage_text() -> str:
    return (
        "✅ Use:\n"
        "• `/predict META 1d`\n"
        "• `/predict META 7d`\n\n"
        f"Allowed: {', '.join(sorted(ALLOWED_TICKERS))}"
    )


# ======================
# CORE ACTION
# ======================
def do_predict(ticker: str, mode: str) -> str:
    days = 1 if mode == "1d" else 7

    preds, last_price, r2, slope, future_dates = predict_prices(ticker, days=days)

    news = fetch_news_yahoo(ticker)
    avg_score, top_items, used_count = analyze_news_semantic(
        news, lookback_days=NEWS_LOOKBACK_DAYS, limit=NEWS_LIMIT
    )
    news_block, news_bias = build_news_reasoning(avg_score, top_items, used_count)

    explanation = build_combined_explanation(slope, news_block, news_bias)

    return build_predict_text(ticker, mode, preds, last_price, r2, future_dates, explanation)


# ======================
# COMMANDS
# ======================
@bot.message_handler(commands=["start"])
def start(message):
    bot.send_message(
        message.chat.id,
        "👋 *Hello!*\n\n"
        "You are using *Predict AI*.\n\n"
        "📌 Commands:\n"
        "• `/predict META 1d`\n"
        "• `/predict META 7d`\n\n"
        "📌 It uses:\n"
        "• Yahoo Finance prices\n"
        "• Yahoo Finance news (via yfinance)\n"
        "• Trend + news keyword scoring\n\n"
        "⚠️ Not financial advice.",
        reply_markup=main_menu(),
        parse_mode="Markdown"
    )


@bot.message_handler(commands=["status"])
def status(message):
    bot.reply_to(
        message,
        f"✅ Bot RUNNING\nSource: Yahoo Finance\nTickers: {', '.join(sorted(ALLOWED_TICKERS))}\n"
        "Command: /predict <TICKER> <1d|7d>"
    )


@bot.message_handler(commands=["predict"])
def predict_command(message):
    ticker, mode = parse_predict_command(message.text)

    if ticker is None and mode is None:
        bot.send_message(message.chat.id, usage_text(), parse_mode="Markdown")
        return

    if ticker not in ALLOWED_TICKERS:
        bot.send_message(message.chat.id, f"Allowed: {', '.join(sorted(ALLOWED_TICKERS))}")
        return

    if mode not in {"1d", "7d"}:
        bot.send_message(message.chat.id, "Mode must be `1d` or `7d`.\n\n" + usage_text(), parse_mode="Markdown")
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
                "Command: /predict <TICKER> <1d|7d>"
            )
            return

        if call.data.startswith("predict_"):
            # predict_META_1d
            _, ticker, mode = call.data.split("_", 2)
            ticker = ticker.upper()
            mode = mode.lower()

            if ticker not in ALLOWED_TICKERS or mode not in {"1d", "7d"}:
                bot.send_message(call.message.chat.id, "Invalid ticker/mode.")
                return

            text = do_predict(ticker, mode)
            bot.send_message(call.message.chat.id, text, parse_mode="Markdown")
            return

    except Exception as e:
        bot.send_message(call.message.chat.id, f"Ошибка: {e}")


# ======================
# RUN
# ======================
if __name__ == "__main__":
    print("Bot started ✅ (Yahoo prices + Yahoo news + news scoring enabled)")
    bot.infinity_polling()
