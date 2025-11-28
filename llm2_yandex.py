# -*- coding: utf-8 -*-
import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------------------
# Загружаем ключи из .env
# -----------------------------------------
load_dotenv()

folder_id = os.getenv("folder_id")
api_key   = os.getenv("api_key")

MODEL = f"gpt://{folder_id}/qwen3-235b-a22b-fp8/latest"

client = OpenAI(
    base_url="https://rest-assistant.api.cloud.yandex.net/v1",
    api_key=api_key,
    project=folder_id
)

# -----------------------------------------
# ТВОИ ВХОДНЫЕ ДАННЫЕ (подставляешь реальные)
# -----------------------------------------
# Пример: ты загружаешь эти три файла после их работы
# Текст письма
with open("letter.txt", "r", encoding="utf-8") as f:
    letter_text = f.read().strip()

# JSON NER
with open("ner_output.json", "r", encoding="utf-8") as f:
    ner_result = json.load(f)

# JSON классификатора
with open("classification_output.json", "r", encoding="utf-8") as f:
    classifier_result = json.load(f)

# JSON LLM1
with open("llm1_output.json", "r", encoding="utf-8") as f:
    llm1_result = json.load(f)

# -----------------------------------------
# PROMPT TEMPLATE (как ты дал)
# -----------------------------------------
PROMPT_TEMPLATE = """Ты — интеллектуальный помощник по подготовке деловой переписки от лица крупного банка.

ТВОЯ РОЛЬ И ПРАВИЛА:
- Ты готовишь ответы на входящие письма клиентов, партнёров и регуляторов.
- Ты пишешь на русском языке, в корректном деловом стиле.
- Ты обязан соблюдать юридическую осторожность.
- У тебя НЕТ доступа к внутренним системам и дополнительным данным, кроме того, что передано ниже.

СТРОГИЕ ЗАПРЕТЫ:
- НЕЛЬЗЯ придумывать суммы, даты, реквизиты договоров, имена, названия подразделений, которых нет во входных данных.
- НЕЛЬЗЯ обещать действия, которые не следуют из данных (например, «мы точно вернём деньги»). Можно только:
  «Банк рассмотрит возможность…», «Банк проведёт проверку…», «Решение будет принято в соответствии с условиями договора и действующим законодательством…».
- НЕЛЬЗЯ добавлять в ответ технические комментарии вроде «как модель ИИ…».

ФОРМАТ ВЫВОДА:
Верни ОДИН валидный JSON следующего вида:

{{
  "answers": {{
    "official": "строгий официальный ответ от лица Банка",
    "business": "деловой ответ, немного менее сухой, чем official, но по сути тот же",
    "client_friendly": "более тёплый, клиентоориентированный ответ без фамильярности",
    "simple": "упрощённый, максимально понятный ответ простым языком"
  }}
}}

--------------------------------
ВХОДНЫЕ ДАННЫЕ
--------------------------------

=== ТЕКСТ ПИСЬМА ===
{letter_text}

=== РЕЗУЛЬТАТ NER (JSON) ===
{ner_json}

Пример структуры NER:
{{
  "contract_numbers": [...],
  "deadlines": [
    {{ "text": "...", "date": "...", "type": "relative|absolute" }}
  ],
  "law_refs": [...],
  "contacts": {{
    "emails": [...],
    "phones": [...],
    "persons": [...]
  }},
  "organizations": [...]
}}

=== РЕЗУЛЬТАТ КЛАССИФИКАТОРА (JSON) ===
{classifier_json}

Пример структуры классификатора:
{{
  "type": "...",
  "urgency": "...",
  "formality": "...",
  "departments": ["...", "..."],
  "legal_risk": "..."
}}

=== РЕЗУЛЬТАТ LLM1 (JSON) ===
{llm1_json}

Пример структуры LLM1:
{{
  "core_request": "...",
  "requirements": "...",
  "expectations": "..."
}}

--------------------------------
ТВОЯ ЗАДАЧА
--------------------------------

1. Проанализируй текст письма с учётом всех входных данных.
2. Подготовь четыре варианта ответа: official, business, client_friendly, simple.
3. Учитывай тип письма и наличие legal_risk.
4. ВЕРНИ СТРОГО JSON такого вида:

{{
  "answers": {{
    "official": "...",
    "business": "...",
    "client_friendly": "...",
    "simple": "..."
  }}
}}
"""

# -----------------------------------------
# Подставляем данные в PROMPT
# -----------------------------------------
prompt = PROMPT_TEMPLATE.format(
    letter_text=letter_text,
    ner_json=json.dumps(ner_result, ensure_ascii=False),
    classifier_json=json.dumps(classifier_result, ensure_ascii=False),
    llm1_json=json.dumps(llm1_result, ensure_ascii=False),
)

# -----------------------------------------
# Вызов Yandex Cloud LLM
# -----------------------------------------
response = client.responses.create(
    model=MODEL,
    input=prompt,
    max_output_tokens=1500,
    temperature=0.1
)

raw = response.output_text.strip()

# -----------------------------------------
# Вырезаем JSON
# -----------------------------------------
if raw.startswith("```"):
    raw = raw.strip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()

start = raw.find("{")
end   = raw.rfind("}")

if start != -1 and end != -1:
    json_str = raw[start:end+1]
else:
    json_str = raw

try:
    result = json.loads(json_str)
except:
    result = {
        "answers": {
            "official": "",
            "business": "",
            "client_friendly": "",
            "simple": ""
        }
    }

# -----------------------------------------
# Сохраняем в JSON файл
# -----------------------------------------
with open("llm2_output.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("Файл сохранён → llm2_output.json")