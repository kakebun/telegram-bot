# bot.py
import os
import re
import telebot
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

# ===== Helpers =====
def get_output_text(resp) -> str:
    """Compatible with different SDK versions: output_text may be str or callable."""
    out = getattr(resp, "output_text", "")
    if callable(out):
        out = out()
    return str(out).strip()

def detect_target_lang(text: str) -> str:
    """
    Decide target language for translation:
    - If text contains Cyrillic and looks Kazakh -> translate to English
    - If text contains Cyrillic and looks Russian -> translate to Kazakh
    - If text is mostly Latin -> translate to Kazakh
    Simple heuristic, good enough for project.
    """
    t = text.strip()
    if not t:
        return "KZ"

    # Cyrillic?
    if re.search(r"[А-Яа-яӘәІіҢңҒғҮүҰұҚқӨөҺһ]", t):
        # Kazakh-specific letters present -> likely KZ -> translate to EN
        if re.search(r"[ӘәІіҢңҒғҮүҰұҚқӨөҺһ]", t):
            return "EN"
        # Otherwise assume Russian -> translate to KZ
        return "KZ"

    # Latin -> translate to KZ
    return "KZ"

def gpt_translate(text: str, target_lang: str) -> str:
    """
    Translate text into target_lang (KZ/EN/RU).
    Output only translation. Kazakh preferred when target_lang=KZ.
    """
    system = (
        "You are a professional translator. "
        "Translate accurately, preserve meaning and tone. "
        "Do NOT add explanations. Output ONLY the translated text."
    )

    prompt = f"Target language: {target_lang}\n\nText:\n{text}"

    resp = client.responses.create(
        model="gpt-5.2",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return get_output_text(resp)

def gpt_assistant_reply(user_text: str) -> str:
    """
    General assistant reply in Kazakh.
    """
    system = (
        "Sen aqyldy, sypaiy, paydaly IІ-komekshisinsin. "
        "Barlyk jauaptardy tek QAZAQ tilinde ber. "
        "Jauaptar anyq, tusinikti, qysqa bolsyn, biraq jetkilikti aqparat bersin."
    )

    resp = client.responses.create(
        model="gpt-5.2",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
    )
    return get_output_text(resp)

def parse_audar_command(text: str):
    """
    If message contains 'аудар' (case-insensitive), return the part after it.
    Examples:
    - "аудар Hello" -> "Hello"
    - "Сәлем, аудар: I love KZ" -> "I love KZ"
    - "аудар, привет" -> "привет"
    If no 'аудар' found or nothing after -> None
    """
    m = re.search(r"\bаудар\b\s*[:\-–—,]?\s*(.+)$", text, flags=re.IGNORECASE)
    if not m:
        return None
    payload = m.group(1).strip()
    return payload if payload else None

# ===== Commands =====
@bot.message_handler(commands=["start"])
def start(msg):
    bot.send_message(
        msg.chat.id,
        "⚠️ *ЕСКЕРТУ (DISCLAIMER):*\n"
        "Бұл бот оқу және эксперименттік мақсатта жасалған.\n"
        "Маңызды заңды/медициналық аудармалар үшін қолданбаңыз.\n\n"
        "👋 *Сәлем!* Мен *QazaqTranslateAI* ботымын.\n"
        "Мен қазақ тілінде жауап беремін және аударма жасай аламын.\n\n"
        "✅ *Қалай қолдану керек:*\n"
        "• Жай сұрақ қой: _«Сәлем, қалайсың?»_ — мен жауап беремін.\n"
        "• Аудару үшін сөйлемнің ішіне *аудар* сөзін жаз:\n"
        "  _«аудар Hello my friend»_ немесе _«Сәлем, аудар: I love KZ»_\n",
        parse_mode="Markdown"
    )

@bot.message_handler(commands=["help"])
def help_cmd(msg):
    bot.send_message(
        msg.chat.id,
        "ℹ️ *Көмек*\n\n"
        "Бот әрқашан қазақ тілінде жауап береді.\n"
        "Аударма қажет болса, мәтіннің алдында немесе ішінде *аудар* деп жаз.\n\n"
        "Мысал:\n"
        "• аудар Hello, how are you?\n"
        "• Сәлем, аудар: Мен бүгін сабаққа бардым.\n",
        parse_mode="Markdown"
    )

# ===== Main handler =====
@bot.message_handler(content_types=["text"])
def handle_text(msg):
    chat_id = msg.chat.id
    text = (msg.text or "").strip()
    if not text:
        return

    try:
        bot.send_chat_action(chat_id, "typing")

        # 1) Translation mode only if 'аудар' exists
        payload = parse_audar_command(text)
        if payload:
            target = detect_target_lang(payload)  # KZ or EN, simple auto
            translated = gpt_translate(payload, target_lang=target)

            if target == "KZ":
                header = "✅ *Міне, сіздің аудармаңыз (қазақша):*\n"
            elif target == "EN":
                header = "✅ *Міне, сіздің аудармаңыз (English):*\n"
            else:
                header = "✅ *Міне, сіздің аудармаңыз:*\n"

            bot.send_message(chat_id, header + translated, parse_mode="Markdown")
            return

        # 2) Otherwise: normal assistant chat in Kazakh
        answer = gpt_assistant_reply(text)
        bot.send_message(chat_id, answer)

    except Exception as e:
        bot.send_message(chat_id, f"⚠️ Қате: {e}")

if __name__ == "__main__":
    print("Bot is running...")
    bot.infinity_polling(skip_pending=True)
