import os
import telebot
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from datetime import datetime, timezone, timedelta
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

# ====== CONFIG ======
API_TOKEN = os.getenv("BOT_TOKEN")  # Railway Variables -> BOT_TOKEN
if not API_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set. Add it in Railway -> Variables.")

bot = telebot.TeleBot(API_TOKEN)

# Yahoo Finance tickers
# META -> META, SNAP -> SNAP, PINS -> PINS
ALLOWED_TICKERS = {"META", "SNAP", "PINS"}

# сколько дней истории брать (чем больше, тем стабильнее регрессия)
HISTORY_PERIOD = "6mo"   # можно: "3mo", "1y"
INTERVAL = "1d"          # дневные свечи

# ====== INLINE MENU ======
def main_menu():
    keyboard = InlineKeyboardMarkup(row_width=2)
    keyboard.add(
        InlineKeyboardButton("📊 Predict META", callback_data="predict_META"),
        InlineKeyboardButton("📊 Predict SNAP", callback_data="predict_SNAP"),
        InlineKeyboardButton("📊 Predict PINS", callback_data="predict_PINS"),
        InlineKeyboardButton("ℹ️ Status", callback_data="status")
    )
    return keyboard


# ====== DATA FROM YAHOO ======
def fetch_close_prices_from_yahoo(ticker: str) -> np.ndarray:
    """
    Берём цены закрытия с Yahoo Finance через yfinance.
    Возвращаем numpy array (N, 1)
    """
    df = yf.download(ticker, period=HISTORY_PERIOD, interval=INTERVAL, progress=False)

    # Иногда yfinance возвращает MultiIndex колонки (редко), приведём к обычному виду
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or "Close" not in df.columns:
        raise ValueError("No data from Yahoo Finance (empty response).")

    close = pd.to_numeric(df["Close"], errors="coerce").dropna().values.reshape(-1, 1)

    if len(close) < 10:
        raise ValueError("Not enough data from Yahoo Finance (need at least 10 days).")

    return close


# ====== MODEL ======
def predict_price_from_yahoo(ticker: str):
    """
    1) качаем Close с Yahoo
    2) обучаем линейную регрессию на индексе времени
    3) предсказываем next close
    """
    close = fetch_close_prices_from_yahoo(ticker)

    X = np.arange(len(close)).reshape(-1, 1)
    y = close

    model = LinearRegression()
    model.fit(X, y)

    predicted = float(model.predict([[len(close)]])[0][0])  # next point
    last_price = float(close[-1][0])
    confidence = float(model.score(X, y))  # R^2

    return predicted, last_price, confidence


# ====== COMMANDS ======
@bot.message_handler(commands=["start"])
def start(message):
    bot.send_message(
        message.chat.id,
        "👋 *Hello!*\n\n"
        "You are using *Predict AI* — a Telegram bot that predicts short-term stock moves.\n\n"
        "📊 *What this bot does now:*\n"
        "• Downloads latest prices directly from *Yahoo Finance*\n"
        "• Uses Linear Regression to detect the trend\n"
        "• Predicts the next closing price\n"
        "• Shows confidence (R²)\n\n"
        "⚠️ *Note:* This is not financial advice.\n\n"
        "Choose an action below 👇",
        reply_markup=main_menu(),
        parse_mode="Markdown"
    )


@bot.message_handler(commands=["status"])
def status(message):
    bot.reply_to(
        message,
        f"✅ Bot RUNNING\nData source: Yahoo Finance\nTickers: {', '.join(sorted(ALLOWED_TICKERS))}"
    )


@bot.message_handler(commands=["predict"])
def predict_command(message):
    parts = message.text.split()
    if len(parts) != 2:
        bot.reply_to(message, "Формат: /predict META")
        return

    ticker = parts[1].upper().strip()
    if ticker not in ALLOWED_TICKERS:
        bot.reply_to(message, f"Разрешены: {', '.join(sorted(ALLOWED_TICKERS))}")
        return

    try:
        predicted, last_price, confidence = predict_price_from_yahoo(ticker)
        direction = "📈 UP" if predicted > last_price else "📉 DOWN"

        explanation = (
            "📊 *Why this move?*\n"
            "• Trend based on Yahoo Finance daily closes\n"
            "• Linear regression continuation\n"
            "• No news/events considered"
        )

        kz_time = datetime.now(timezone.utc) + timedelta(hours=5)

        text = (
            f"*{ticker} Prediction (Yahoo Finance)*\n\n"
            f"Last close: `{last_price:.2f}`\n"
            f"Predicted next close: `{predicted:.2f}`\n"
            f"Direction: {direction}\n"
            f"Confidence (R²): `{confidence*100:.1f}%`\n\n"
            f"{explanation}\n\n"
            f"⏱ {kz_time.strftime('%Y-%m-%d %H:%M')}"
        )

        bot.send_message(message.chat.id, text, parse_mode="Markdown")

    except Exception as e:
        bot.reply_to(message, f"Ошибка: {e}")


# ====== BUTTON HANDLER ======
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    if call.data.startswith("predict_"):
        ticker = call.data.split("_")[1].upper()

        if ticker not in ALLOWED_TICKERS:
            bot.send_message(call.message.chat.id, "Ticker not allowed.")
            return

        try:
            predicted, last_price, confidence = predict_price_from_yahoo(ticker)
            direction = "📈 UP" if predicted > last_price else "📉 DOWN"

            explanation = (
                "📊 *Why this move?*\n"
                "• Trend based on Yahoo Finance daily closes\n"
                "• Linear regression continuation\n"
                "• No news/events considered"
            )

            kz_time = datetime.now(timezone.utc) + timedelta(hours=5)

            text = (
                f"*{ticker} Prediction (Yahoo Finance)*\n\n"
                f"Last close: `{last_price:.2f}`\n"
                f"Predicted next close: `{predicted:.2f}`\n"
                f"Direction: {direction}\n"
                f"Confidence (R²): `{confidence*100:.1f}%`\n\n"
                f"{explanation}\n\n"
                f"⏱ {kz_time.strftime('%Y-%m-%d %H:%M')}"
            )

            bot.send_message(call.message.chat.id, text, parse_mode="Markdown")

        except Exception as e:
            bot.send_message(call.message.chat.id, f"Ошибка: {e}")

    elif call.data == "status":
        bot.send_message(
            call.message.chat.id,
            f"✅ Bot RUNNING\nData source: Yahoo Finance\nTickers: {', '.join(sorted(ALLOWED_TICKERS))}"
        )


# ====== RUN ======
if __name__ == "__main__":
    print("Bot started with INLINE MENU ✅ (Yahoo Finance enabled)")
    bot.infinity_polling()
