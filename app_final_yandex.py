import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# ===========================================
# Загружаем ключи
# ===========================================
load_dotenv()

folder_id = os.getenv("folder_id")
api_key   = os.getenv("api_key")

MODEL = f"gpt://{folder_id}/qwen3-235b-a22b-fp8/latest"

client = OpenAI(
    base_url="https://rest-assistant.api.cloud.yandex.net/v1",
    api_key=api_key,
    project=folder_id
)

# ===========================================
# ВХОД: Просто строка
# ===========================================
# letter_text = """
# Настоящим уведомляем о грубом нарушении условий договора №БС-1456 от 15.03.2023.
# Средства с нашего счёта были списаны без предварительного уведомления.
# Просим предоставить разъяснения.
# """  # ← Можешь менять или подгружать из файла

# Загрузка текста письма
with open("letter.txt", "r", encoding="utf-8") as f:
    letter_text = f.read().strip()

# ===========================================
# Параметры классификатора
# ===========================================
TYPE_CATEGORIES = [
    "Запрос информации/документов",
    "Официальная жалоба или претензия",
    "Регуляторный запрос",
    "Партнёрское предложение",
    "Запрос на согласование",
    "Уведомление или информирование",
]

DEPARTMENT_CATEGORIES_EXPANDED = [

    # --- Юридический блок: детализация ---
    "Юридический блок / Договорное право",
    "Юридический блок / Корпоративное право",
    "Юридический блок / Банковское и финансовое право",
    "Юридический блок / Трудовое право",
    "Юридический блок / Семейное право",
    "Юридический блок / Налогово-правовые вопросы",
    "Юридический блок / Судебно-претензионная работа",
    "Юридический блок / Регуляторика и взаимодействие с ЦБ",
    "Юридический блок / Антимонопольное право и конкуренция",
    "Юридический блок / Интеллектуальная собственность и лицензирование",
    "Юридический блок / Договоры с клиентами и контрагентами",
    "Юридический блок / Корпоративное управление и органы банка",
]


def build_prompt(text: str) -> str:
    categories = "\n".join(f"- {c}" for c in TYPE_CATEGORIES)
    deps       = "\n".join(f"- {d}" for d in DEPARTMENT_CATEGORIES)

    return f"""
Классифицируй входящее письмо.

1) type — одна из категорий:
{categories}

2) urgency — одна из категорий:
• очень срочно
• средне срочно
• не срочно

3) formality — официальный / неофициальный

4) departments — массив подразделений ИЗ списка:
{deps}

5) legal_risk — есть/нет

Формат ответа строго JSON:
{{
 "type": "...",
 "urgency": "...",
 "formality": "...",
 "departments": ["...", "..."],
 "legal_risk": "..."
}}

ТЕКСТ:
{text}
""".strip()

# ===========================================
# Основная функция
# ===========================================
def classify_letter(text: str) -> dict:
    prompt = build_prompt(text)

    response = client.responses.create(
        model=MODEL,
        input=prompt,
        max_output_tokens=300,
        temperature=0.0
    )

    raw = response.output_text.strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()

    try:
        return json.loads(raw)
    except Exception:
        return {
            "type": "Уведомление или информирование",
            "urgency": "не срочно",
            "formality": "официальный",
            "departments": [],
            "legal_risk": "нет"
        }

# ===========================================
# Запуск
# ===========================================
result = classify_letter(letter_text)

print(json.dumps(result, ensure_ascii=False, indent=2))

with open("classification_output.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("Файл сохранён → classification_output.json")