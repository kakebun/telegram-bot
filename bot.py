# bot.py
import os
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from openai import OpenAI

# ====== ENV (Railway Variables) ======
BOT_TOKEN = os.getenv("BOT_TOKEN")          # <-- как у тебя в Railway
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set. Add it in Railway Variables (BOT_TOKEN).")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Add it in Railway Variables (OPENAI_API_KEY).")

bot = telebot.TeleBot(BOT_TOKEN)
client = OpenAI(api_key=OPENAI_API_KEY)

# ====== User state (direction per user) ======
# chat_id -> ("EN","KZ") etc
user_direction = {}

LANG_NAME = {
    "EN": "English",
    "KZ": "Kazakh",
    "RU": "Russian"
}

def main_menu():
    kb = InlineKeyboardMarkup(row_width=2)
    kb.add(
        InlineKeyboardButton("EN → KZ", callback_data="DIR_EN_KZ"),
        InlineKeyboardButton("KZ → EN", callback_data="DIR_KZ_EN"),
        InlineKeyboardButton("RU → KZ", callback_data="DIR_RU_KZ"),
        InlineKeyboardButton("KZ → RU", callback_data="DIR_KZ_RU"),
        InlineKeyboardButton("EN → RU", callback_data="DIR_EN_RU"),
        InlineKeyboardButton("RU → EN", callback_data="DIR_RU_EN"),
    )
    kb.add(InlineKeyboardButton("ℹ️ Help", callback_data="HELP"))
    return kb

def translate_with_gpt(text: str, src: str, dst: str) -> str:
    system = (
        "You are a professional translator. "
        "Translate accurately, preserve meaning, tone, and formatting. "
        "Do NOT add explanations. Output ONLY the translated text."
    )

    prompt = (
        f"Source language: {LANG_NAME[src]}\n"
        f"Target language: {LANG_NAME[dst]}\n\n"
        f"Text:\n{text}"
    )

    resp = client.responses.create(
        model="gpt-5.2",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )

    # robust output parsing
    out = getattr(resp, "output_text", None)
    if callable(out):
        return out().strip()
    if isinstance(out, str):
        return out.strip()
    try:
        return resp.output[0].content[0].text.strip()
    except Exception:
        return "⚠️ Error: could not read model output."

@bot.message_handler(commands=["start"])
def start(message):
    chat_id = message.chat.id
    user_direction[chat_id] = ("EN", "KZ")  # default

    bot.send_message(
        chat_id,
        "⚠️ DISCLAIMER:\n"
        "This bot is for educational/experimental purposes.\n"
        "Do not rely on it for critical/legal/medical translations.\n\n"
        "Choose translation direction:",
        reply_markup=main_menu()
    )

@bot.callback_query_handler(func=lambda call: True)
def on_callback(call):
    chat_id = call.message.chat.id
    data = call.data

    if data == "HELP":
        src, dst = user_direction.get(chat_id, ("EN", "KZ"))
        bot.answer_callback_query(call.id)
        bot.send_message(
            chat_id,
            f"Current direction: {src} → {dst}\n"
            "Send me any text, and I will translate it.\n"
            "Use buttons to change direction.",
            reply_markup=main_menu()
        )
        return

    if data.startswith("DIR_"):
        parts = data.split("_")
        # DIR_EN_KZ -> ["DIR","EN","KZ"]
        if len(parts) == 3:
            src, dst = parts[1], parts[2]
            user_direction[chat_id] = (src, dst)
            bot.answer_callback_query(call.id, f"Direction set: {src} → {dst}")
            bot.send_message(
                chat_id,
                f"✅ Now translating: {src} → {dst}\nSend text:",
                reply_markup=main_menu()
            )
            return

    bot.answer_callback_query(call.id, "Unknown action")

@bot.message_handler(content_types=["text"])
def on_text(message):
    chat_id = message.chat.id
    text = message.text.strip()

    if not text:
        return

    src, dst = user_direction.get(chat_id, ("EN", "KZ"))

    try:
        bot.send_chat_action(chat_id, "typing")
        translated = translate_with_gpt(text, src, dst)

        # Telegram safety length
        if len(translated) > 3500:
            translated = translated[:3500] + "\n\n…(cut)"

        bot.send_message(chat_id, translated, reply_markup=main_menu())

    except Exception as e:
        bot.send_message(chat_id, f"⚠️ Error: {e}")

if __name__ == "__main__":
    print("Bot is running...")
    bot.infinity_polling(skip_pending=True)
