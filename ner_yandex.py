# -*- coding: utf-8 -*-
import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Загружаем ключи и проект
# -----------------------------
load_dotenv()

folder_id = os.getenv("folder_id")
api_key   = os.getenv("api_key")

MODEL = f"gpt://{folder_id}/qwen3-235b-a22b-fp8/latest"

client = OpenAI(
    base_url="https://rest-assistant.api.cloud.yandex.net/v1",
    api_key=api_key,
    project=folder_id
)

# -----------------------------
# ВХОДНОЙ ТЕКСТ
# -----------------------------
# letter_text = """
# Настоящим уведомляем о грубом нарушении условий договора №БС-1456 от 15.03.2023:
# средства с нашего счёта были списаны без предварительного уведомления.
# Требуем немедленного разъяснения и возврата средств в течение 3 рабочих дней,
# но не позднее 20.03.2023.

# На основании п. 4.2 Указания Банка России №55-У от 10.04.2024 просим представить информацию
# о сделках с признаками возможного отмывания денежных средств за III квартал 2025 года.

# Контактное лицо: Иванов И.И., ПАО «Банк Пример»,
# e-mail: complaints@bank-prim.ru, телефон: +7 (495) 123-45-67.
# """
# Загрузка текста письма
with open("letter.txt", "r", encoding="utf-8") as f:
    letter_text = f.read().strip()

# -----------------------------
# PROMPT (system + user)
# -----------------------------
system_prompt = (
    "Ты — нейросетевая модель для извлечения структурированной информации (NER) "
    "из деловой переписки на русском языке.\n\n"
    "Твоя задача — по тексту письма вернуть JSON с полями:\n"
    "- contract_numbers: список строк с номерами договоров/соглашений.\n"
    "- deadlines: список объектов { \"text\": ..., \"date\": ..., \"type\": ... }.\n"
    "- law_refs: список строк с упоминаниями нормативных актов.\n"
    "- contacts: объект { \"emails\": [...], \"phones\": [...], \"persons\": [...] }.\n"
    "- organizations: список строк с названиями организаций.\n\n"
    "Требования:\n"
    "- Если информации нет — верни пустые поля.\n"
    "- Ничего не придумывай.\n"
    "- Ответ ДОЛЖЕН быть валидным JSON без комментариев."
)

user_prompt = (
    "Текст письма:\n\n"
    f"\"\"\"{letter_text}\"\"\"\n\n"
    "Верни строго JSON:\n\n"
    "{\n"
    "  \"contract_numbers\": [ \"...\" ],\n"
    "  \"deadlines\": [\n"
    "    {\n"
    "      \"text\": \"...\",\n"
    "      \"date\": \"... или null\",\n"
    "      \"type\": \"relative или absolute\"\n"
    "    }\n"
    "  ],\n"
    "  \"law_refs\": [ \"...\" ],\n"
    "  \"contacts\": {\n"
    "    \"emails\": [ \"...\" ],\n"
    "    \"phones\": [ \"...\" ],\n"
    "    \"persons\": [ \"...\" ]\n"
    "  },\n"
    "  \"organizations\": [ \"...\" ]\n"
    "}\n\n"
    "Никакого текста вне JSON."
)

# -----------------------------
# ВЫЗОВ YANDEX CLOUD LLM
# -----------------------------
response = client.responses.create(
    model=MODEL,
    input=[{"role": "system", "content": system_prompt},
           {"role": "user",   "content": user_prompt}],
    max_output_tokens=700,
    temperature=0.0
)

raw = response.output_text.strip()

# -----------------------------
# ВЫРЕЗАЕМ JSON
# -----------------------------
if raw.startswith("```"):
    raw = raw.strip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()

start = raw.find("{")
end = raw.rfind("}")
if start != -1 and end != -1:
    raw_json = raw[start:end+1]
else:
    raw_json = raw

data = json.loads(raw_json)

# -----------------------------
# НОРМАЛИЗАЦИЯ ПОЛЕЙ
# -----------------------------
def safe_list(x):
    return x if isinstance(x, list) else []

def safe_obj(x):
    return x if isinstance(x, dict) else {}

contract_numbers = safe_list(data.get("contract_numbers"))
law_refs         = safe_list(data.get("law_refs"))
organizations    = safe_list(data.get("organizations"))
contacts         = safe_obj(data.get("contacts"))
deadlines_raw    = safe_list(data.get("deadlines"))

emails  = safe_list(contacts.get("emails"))
phones  = safe_list(contacts.get("phones"))
persons = safe_list(contacts.get("persons"))

deadlines = []
for d in deadlines_raw:
    if isinstance(d, dict):
        deadlines.append({
            "text": str(d.get("text", "")),
            "date": d.get("date", None),
            "type": str(d.get("type", ""))
        })

result = {
    "contract_numbers": [str(x) for x in contract_numbers],
    "deadlines": deadlines,
    "law_refs": [str(x) for x in law_refs],
    "contacts": {
        "emails": [str(x) for x in emails],
        "phones": [str(x) for x in phones],
        "persons": [str(x) for x in persons],
    },
    "organizations": [str(x) for x in organizations],
}

# -----------------------------
# СОХРАНЯЕМ В JSON ФАЙЛ
# -----------------------------
with open("ner_output.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("Файл сохранён → ner_output.json")