import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

# ------------------------
# Загружаем API ключи
# ------------------------
load_dotenv()

folder_id = os.getenv("folder_id")
api_key   = os.getenv("api_key")

MODEL = f"gpt://{folder_id}/qwen3-235b-a22b-fp8/latest"

client = OpenAI(
    base_url="https://rest-assistant.api.cloud.yandex.net/v1",
    api_key=api_key,
    project=folder_id
)

# ------------------------
# Входной текст
# ------------------------
# letter_text = """
# Настоящим уведомляем о грубом нарушении условий договора №БС-1456 от 15.03.2023: средства с нашего счёта были списаны без предварительного уведомления. Требуем немедленного разъяснения и возврата средств в течение 3 рабочих дней.
# """
# Загрузка текста письма
with open("letter.txt", "r", encoding="utf-8") as f:
    letter_text = f.read().strip()
# ------------------------
# PROMPT
# ------------------------
prompt = f"""
Ты — эксперт по анализу официальной деловой переписки.

Тебе даётся текст одного письма/запроса на русском языке.
Твоя задача — кратко и структурировано выделить из него:

1) Суть запроса — в одном-двух предложениях: о чём письмо и чего в целом хочет отправитель.
2) Требования — конкретные действия, которые должен выполнить адресат (включая сроки, суммы, документы и т.п.).
3) Ожидания отправителя — что отправитель рассчитывает получить по итогам выполнения требований (результат, эффект, статус).

Важно:
- Пиши по-русски.
- Отвечай строго в формате JSON.
- Не добавляй никаких пояснений вне JSON.
- Если какой-то части явно нет в тексте, пиши туда null.

Формат ответа:

{{
  "core_request": "краткое описание сути запроса",
  "requirements": "список или сжатое описание конкретных требований",
  "expectations": "чего отправитель ожидает по результату"
}}

ТЕКСТ ПИСЬМА:
---
{letter_text}
---
""".strip()

# ------------------------
# Вызов Яндекс LLM
# ------------------------
response = client.responses.create(
    model=MODEL,
    input=prompt,
    max_output_tokens=300,
    temperature=0.0
)

raw = response.output_text.strip()

# ------------------------
# Чистим ````json обёртки
# ------------------------
text = raw

if text.startswith("```"):
    parts = text.split("```")
    if len(parts) >= 2:
        text = parts[1].strip()
        first_newline = text.find("\n")
        if first_newline != -1 and not text.lstrip().startswith("{"):
            maybe_lang = text[:first_newline].lower()
            if "json" in maybe_lang:
                text = text[first_newline + 1:].strip()

# ------------------------
# Ищем JSON-объекты
# ------------------------
candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
target_keys = {"core_request", "requirements", "expectations"}

valid_objects = []
for js in candidates:
    try:
        obj = json.loads(js)
        valid_objects.append(obj)
    except:
        continue

best_obj = None
if valid_objects:
    for obj in valid_objects:
        if any(k in obj for k in target_keys):
            best_obj = obj
            break
    if best_obj is None:
        best_obj = valid_objects[0]

# ------------------------
# Если JSON не найден
# ------------------------
if best_obj is None:
    best_obj = {
        "core_request": None,
        "requirements": None,
        "expectations": None
    }

# ------------------------
# Сохраняем в JSON файл
# ------------------------
with open("llm1_output.json", "w", encoding="utf-8") as f:
    json.dump(best_obj, f, ensure_ascii=False, indent=2)

print("Файл сохранён → llm1_output.json")