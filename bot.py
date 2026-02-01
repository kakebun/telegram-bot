# bot.py
import os
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from openai import OpenAI

# ===== ENV =====
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

bot = telebot.TeleBot(BOT_TOKEN)
client = OpenAI(api_key=OPENAI_API_KEY)

# ===== USER STATE =====
# chat_id -> dict
users = {}

# ===== MENUS =====
def main_menu():
    kb = InlineKeyboardMarkup(row_width=2)
    kb.add(
        InlineKeyboardButton("🔵 Аударма", callback_data="MODE_TRANSLATE"),
        InlineKeyboardButton("🟢 Әңгіме", callback_data="MODE_CHAT"),
    )
    kb.add(InlineKeyboardButton("ℹ️ Көмек", callback_data="HELP"))
    return kb

def translate_menu():
    kb = InlineKeyboardMarkup(row_width=2)
    kb.add(
        InlineKeyboardButton("EN → KZ", callback_data="DIR_EN_KZ"),
        InlineKeyboardButton("KZ → EN", callback_data="DIR_KZ_EN"),
        InlineKeyboardButton("RU → KZ", callback_data="DIR_RU_KZ"),
        InlineKeyboardButton("KZ → RU", callback_data="DIR_KZ_RU"),
        InlineKeyboardButton("EN → RU", callback_data="DIR_EN_RU"),
        InlineKeyboardButton("RU → EN", callback_data="DIR_RU_EN"),
    )
    kb.add(InlineKeyboardButton("⬅️ Артқа", callback_data="BACK"))
    return kb

# ===== GPT FUNCTIONS =====
def gpt_translate(text, src, dst):
    system = (
        "You are a professional translator. "
        "Translate accurately. Do not add explanations."
    )
    prompt = f"Translate from {src} to {dst}:\n{text}"

    r = client.responses.create(
        model="gpt-5.2",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return r.output_text().strip()

def gpt_chat(text):
    system = (
        "Sen aqылды, сыпайы көмекші ИИ-сің. "
        "Барлық жауаптарды ҚАЗАҚ тілінде бер. "
        "Жауаптар түсінікті, нақты және достық стильде болсын."
    )

    r = client.responses.create(
        model="gpt-5.2",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
    )
    return r.output_text().strip()

# ===== COMMANDS =====
@bot.message_handler(commands=["start"])
def start(msg):
    chat_id = msg.chat.id
    users[chat_id] = {
        "mode": "CHAT",
        "dir": ("EN", "KZ")
    }

    bot.send_message(
        chat_id,
        "⚠️ *ЕСКЕРТУ (DISCLAIMER):*\n"
        "Бұл бот оқу және эксперименттік мақсатта жасалған.\n"
        "Заңды, медициналық немесе маңызды аудармалар үшін қолданбаңыз.\n\n"
        "👋 *Сәлем!* Мен *QazaqTranslateAI* ботымын.\n"
        "Мен сенімен қазақ тілінде сөйлесе аламын және мәтіндерді аудара аламын.\n\n"
        "Режимді таңда:",
        parse_mode="Markdown",
        reply_markup=main_menu()
    )

# ===== CALLBACKS =====
@bot.callback_query_handler(func=lambda c: True)
def callbacks(call):
    chat_id = call.message.chat.id
    data = call.data

    if data == "MODE_CHAT":
        users[chat_id]["mode"] = "CHAT"
        bot.send_message(chat_id, "🟢 Әңгіме режимі қосылды. Маған жаза бер 🙂")

    elif data == "MODE_TRANSLATE":
        users[chat_id]["mode"] = "TRANSLATE"
        bot.send_message(chat_id, "🔵 Аударма режимі. Бағытты таңда:", reply_markup=translate_menu())

    elif data.startswith("DIR_"):
        _, src, dst = data.split("_")
        users[chat_id]["dir"] = (src, dst)
        bot.send_message(chat_id, f"✅ Аударма бағыты: {src} → {dst}\nМәтін жібер.")

    elif data == "HELP":
        bot.send_message(
            chat_id,
            "ℹ️ *Көмек*\n\n"
            "🟢 Әңгіме – ботпен қазақ тілінде сөйлесу\n"
            "🔵 Аударма – тілдер арасында аудару\n\n"
            "Режимді батырмалар арқылы таңда.",
            parse_mode="Markdown"
        )

    elif data == "BACK":
        bot.send_message(chat_id, "Басты мәзір:", reply_markup=main_menu())

# ===== TEXT HANDLER =====
@bot.message_handler(content_types=["text"])
def handle_text(msg):
    chat_id = msg.chat.id
    text = msg.text.strip()

    mode = users.get(chat_id, {}).get("mode", "CHAT")

    try:
        bot.send_chat_action(chat_id, "typing")

        if mode == "CHAT":
            answer = gpt_chat(text)
        else:
            src, dst = users[chat_id]["dir"]
            answer = gpt_translate(text, src, dst)

        bot.send_message(chat_id, answer)

    except Exception as e:
        bot.send_message(chat_id, f"⚠️ Қате: {e}")

# ===== RUN =====
print("Bot started...")
bot.infinity_polling(skip_pending=True)
