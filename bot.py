import telebot
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timezone, timedelta
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

import os
API_TOKEN = os.getenv("BOT_TOKEN")

bot = telebot.TeleBot(API_TOKEN)

ALLOWED_TICKERS = {"META", "SNAP", "PINS"}

# ---------- INLINE MENU ----------
def main_menu():
    keyboard = InlineKeyboardMarkup(row_width=2)
    keyboard.add(
        InlineKeyboardButton("📊 Predict META", callback_data="predict_META"),
        InlineKeyboardButton("📊 Predict SNAP", callback_data="predict_SNAP"),
        InlineKeyboardButton("📊 Predict PINS", callback_data="predict_PINS"),
        InlineKeyboardButton("ℹ️ Status", callback_data="status")
    )
    return keyboard

# ---------- MODEL ----------
def predict_price_from_csv(ticker: str):
    df = pd.read_csv(f"{ticker}.csv")

    df = df[["Close"]]
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna()

    close = df["Close"].values.reshape(-1, 1)
    if len(close) < 5:
        raise ValueError("Not enough numeric data")

    X = np.arange(len(close)).reshape(-1, 1)
    y = close

    model = LinearRegression()
    model.fit(X, y)

    predicted = float(model.predict([[len(close)]])[0][0])
    last_price = float(close[-1][0])
    confidence = float(model.score(X, y))

    return predicted, last_price, confidence

# ---------- COMMANDS ----------
@bot.message_handler(commands=["start"])
def start(message):
    bot.send_message(
        message.chat.id,
        "👋 *Hello!*\n\n"
        "You are using *Predict AI* — an AI-powered Telegram bot that helps "
        "predict short-term stock price movements based on historical data.\n\n"
        "📊 *What this bot does:*\n"
        "• Analyzes past prices (CSV data)\n"
        "• Uses Linear Regression to detect trends\n"
        "• Predicts the next closing price\n"
        "• Shows confidence of the prediction\n\n"
        "⚠️ *Note:* This is not financial advice.\n\n"
        "Choose an action below 👇",
        reply_markup=main_menu(),
        parse_mode="Markdown"
    )

@bot.message_handler(commands=["status"])
def status(message):
    bot.reply_to(
        message,
        f"✅ Bot RUNNING\nTickers: {', '.join(sorted(ALLOWED_TICKERS))}"
    )

# ✅ ДОБАВЛЕНО: команда /predict META
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
        predicted, last_price, confidence = predict_price_from_csv(ticker)
        direction = "📈 UP" if predicted > last_price else "📉 DOWN"

        explanation = (
            "📊 *Why this move?*\n"
            "• Trend based on last prices\n"
            "• Linear regression continuation\n"
            "• No news/events considered"
        )

        kz_time = datetime.now(timezone.utc) + timedelta(hours=5)

        text = (
            f"*{ticker} Prediction*\n\n"
            f"Last close: `{last_price:.2f}`\n"
            f"Predicted close: `{predicted:.2f}`\n"
            f"Direction: {direction}\n"
            f"Confidence (R²): `{confidence*100:.1f}%`\n\n"
            f"{explanation}\n\n"
            f"⏱ {kz_time.strftime('%Y-%m-%d %H:%M')}"
        )

        bot.send_message(message.chat.id, text, parse_mode="Markdown")

    except Exception as e:
        bot.reply_to(message, f"Ошибка: {e}")

# ---------- BUTTON HANDLER ----------
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    if call.data.startswith("predict_"):
        ticker = call.data.split("_")[1]

        try:
            predicted, last_price, confidence = predict_price_from_csv(ticker)
            direction = "📈 UP" if predicted > last_price else "📉 DOWN"

            explanation = (
                "📊 *Why this move?*\n"
                "• Trend based on last prices\n"
                "• Linear regression continuation\n"
                "• No news/events considered"
            )

            kz_time = datetime.now(timezone.utc) + timedelta(hours=5)

            text = (
                f"*{ticker} Prediction*\n\n"
                f"Last close: `{last_price:.2f}`\n"
                f"Predicted close: `{predicted:.2f}`\n"
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
            f"✅ Bot RUNNING\nTickers: {', '.join(sorted(ALLOWED_TICKERS))}"
        )

# ---------- RUN ----------
if __name__ == "__main__":
    print("Bot started with INLINE MENU ✅")
    bot.infinity_polling()
