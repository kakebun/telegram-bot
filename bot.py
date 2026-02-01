# bot.py
import os
import re
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from openai import OpenAI

# ===== ENV (Railway Variables) =====
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set. Add it in Railway Variables (BOT_TOKEN).")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Add it in Railway Variables (OPENAI_API_KEY).")

bot = telebot.TeleBot(BOT_TOKEN)
client = OpenAI(api_key=OPENAI_API_KEY)

# ===== USER STATE =====
# chat_id -> {"src": "EN", "dst": "KZ"}
users = {}

# ===== Translation directions =====
DIRS = [
    ("EN", "KZ"),
    ("KZ", "EN"),
    ("RU", "KZ"),
    ("KZ", "RU"),
    ("EN", "RU"),
    ("RU", "EN"),
]

def dir_menu(current_src="EN", current_dst="KZ"):
    kb = InlineKeyboardMarkup(row_width=2)
    for src, dst in DIRS:
        label = f"{src} → {dst}"
        if src == current_src and dst == current_dst:
            label = "✅ " + label
        kb.add(InlineKeyboardButton(label, callback_data=f"DIR_{src}_{dst}"))
    kb.add(InlineKeyboardButton("ℹ️ Көмек", callback_data="HELP"))
    return kb

# ===== Helpers =====
def get_output_text(resp) -> str:
    out = getattr(resp, "output_text", "")
    if callable(out):
        out = out()
    return str(out).strip()

def parse_audar_command(text: str):
    # "аудар Hello" OR "Сәлем, аудар: Hello"
    m = re.search(r"\bаудар\b\s*[:\-–—,]?\s*(.+)$", text, flags=re.IGNORECASE)
    if not m:
        return None
    payload = m.group(1).strip()
    return payload if payload else None

def is_thanks(text: str) -> bool:
    t = text.lower()
    keys = ["рақмет", "рахмет", "спасибо", "thank", "thx"]
    return any(k in t for k in keys)

# ===== OpenAI calls =====
def assistant_reply_kz(user_text: str) -> str:
    system = (
        "Sen qazaq tilinde jauap beretin AI-komekshisinsin. "
        "Jauap qysqa jane naqty bolsyn (1-3 jūyeler). "
        "Eshqandai Markdown qoldanba: *, **, #, •, тізімдер, код блоктар жоқ. "
        "Adam 'tusindir', 'nеге', 'толық' dep suramasa — analiz/tusindirme jasama."
    )
    resp = client.responses.create(
        model="gpt-5.2",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
    )
    return get_output_text(resp)

def translate_text(text: str, src: str, dst: str) -> str:
    system = (
        "You are a professional translator. "
        "Translate accurately, preserve meaning and tone. "
        "Do NOT add explanations. Output ONLY the translated text. "
        "No Markdown, no asterisks, no bullet lists."
    )
    prompt = f"Source language: {src}\nTarget language: {dst}\n\nText:\n{text}"
    resp = client.responses.create(
        model="gpt-5.2",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return get_output_text(resp)

# ===== Commands =====
@bot.message_handler(commands=["start"])
def start(msg):
    chat_id = msg.chat.id
    users[chat_id] = {"src": "EN", "dst": "KZ"}

    text = (
        "⚠️ ЕСКЕРТУ (DISCLAIMER):\n"
        "Бұл бот оқу және эксперименттік мақсатта жасалған.\n"
        "Маңызды заңды/медициналық аудармалар үшін қолданбаңыз.\n\n"
        "👋 Сәлем! Мен QazaqTranslateAI ботымын.\n\n"
        "🔤 Аудару үшін хабарламада 'аудар' сөзін қолданыңыз:\n"
        "Мысал: аудар Hello my friend\n"
        "немесе: Сәлем, аудар: I love KZ\n\n"
        "🌍 Тілді таңдаңыз (қай тілден → қай тілге):"
    )
    bot.send_message(chat_id, text, reply_markup=dir_menu("EN", "KZ"))

@bot.message_handler(commands=["тiл"])
def choose_language(msg):
    chat_id = msg.chat.id
    if chat_id not in users:
        users[chat_id] = {"src": "EN", "dst": "KZ"}

    src = users[chat_id]["src"]
    dst = users[chat_id]["dst"]

    bot.send_message(
        chat_id,
        "🌍 Аударма бағытын таңдаңыз (қай тілден → қай тілге):",
        reply_markup=dir_menu(src, dst)
    )

# ===== Callback buttons =====
@bot.callback_query_handler(func=lambda c: True)
def callbacks(call):
    chat_id = call.message.chat.id
    data = call.data

    if chat_id not in users:
        users[chat_id] = {"src": "EN", "dst": "KZ"}

    if data.startswith("DIR_"):
        _, src, dst = data.split("_", 2)
        users[chat_id]["src"] = src
        users[chat_id]["dst"] = dst

        bot.answer_callback_query(call.id, f"Таңдалды: {src} → {dst}")

        # After choosing language: confirm WITHOUT menu spam
        bot.send_message(chat_id, f"✅ Жақсы! Қазір бағыт: {src} → {dst}\n✍️ Аудару үшін: аудар ... деп жазыңыз.")
        return

    if data == "HELP":
        src = users[chat_id]["src"]
        dst = users[chat_id]["dst"]
        bot.answer_callback_query(call.id)

        bot.send_message(
            chat_id,
            "ℹ️ Көмек:\n"
            "• Тілді өзгерту үшін: /тiл\n"
            "• Аудару үшін: аудар + мәтін\n"
            f"Қазіргі бағыт: {src} → {dst}"
        )
        return

    bot.answer_callback_query(call.id, "OK")

# ===== Main messages =====
@bot.message_handler(content_types=["text"])
def handle_text(msg):
    chat_id = msg.chat.id
    text = (msg.text or "").strip()
    if not text:
        return

    if chat_id not in users:
        users[chat_id] = {"src": "EN", "dst": "KZ"}

    src = users[chat_id]["src"]
    dst = users[chat_id]["dst"]

    try:
        bot.send_chat_action(chat_id, "typing")

        # thanks -> fixed reply
        if is_thanks(text) and "аудар" not in text.lower():
            bot.send_message(chat_id, "Әрқашан көмектесуге дайынмын 😊")
            return

        # translation only when "аудар"
        payload = parse_audar_command(text)
        if payload:
            translated = translate_text(payload, src, dst)
            bot.send_message(chat_id, f"✅ Міне, сіздің аудармаңыз ({src} → {dst}):\n{translated}")
            return

        # normal assistant reply
        answer = assistant_reply_kz(text)
        bot.send_message(chat_id, answer)

    except Exception as e:
        bot.send_message(chat_id, f"⚠️ Қате: {e}")

if __name__ == "__main__":
    print("Bot is running...")
    bot.infinity_polling(skip_pending=True)
