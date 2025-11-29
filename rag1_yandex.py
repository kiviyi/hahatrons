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
# Входные данные
# ------------------------

# Текст письма
with open("letter.txt", "r", encoding="utf-8") as f:
    letter_text = f.read().strip()

# NER
with open("ner_output.json", "r", encoding="utf-8") as f:
    ner_result = json.load(f)

# Классификатор
with open("classification_output.json", "r", encoding="utf-8") as f:
    classifier_result = json.load(f)

# LLM1 (core_request / requirements / expectations)
with open("llm1_output.json", "r", encoding="utf-8") as f:
    llm1_result = json.load(f)

ner_json        = json.dumps(ner_result, ensure_ascii=False, indent=2)
classifier_json = json.dumps(classifier_result, ensure_ascii=False, indent=2)
llm1_json       = json.dumps(llm1_result, ensure_ascii=False, indent=2)

# ------------------------
# PROMPT: подбор документов
# ------------------------
prompt = f"""
Ты — эксперт по внутренним документам крупного банка (регламенты, методики, НПА, шаблоны писем, кейсы переписки).

ТВОЯ ЗАДАЧА:
По входным данным о письме клиента и его классификации:
- предположить, КАКИЕ типы документов банк обычно использует в таких ситуациях;
- вернуть СТРУКТУРИРОВАННЫЙ список:
  - "valid_docs" — нормативные и внутренние документы, на которые можно ссылаться (НПА, внутренние регламенты, методики, стандарты);
  - "recommended_docs" — рекомендованные материалы (шаблоны писем, примеры кейсов, внутренние рекомендации).

ВАЖНО:
- Ты НЕ знаешь точного содержания документов банка и не можешь придумывать реальные номера и названия НПА или внутренних документов.
- Вместо этого ты описываешь документы ТИПАМИ и ЧЕЛОВЕЧЕСКИ-ПОНЯТНЫМИ ОБОЗНАЧЕНИЯМИ, чтобы их потом можно было сопоставить с реальными документами в системе Банка.
- НЕ нужно выдумывать конкретные номера указаний, приказов и т.п., если их нет во входных данных. Если номер/название НПА есть во входных данных (NER.law_refs), можно его использовать.

ФОРМАТ ВЫВОДА:
Верни ОДИН валидный JSON вида:

{{
  "valid_docs": [
    {{
      "code": "string, короткий код документа (например, 'POLICY_COMPLAINTS', 'LAW_FROM_LETTER')",
      "kind": "law | policy | methodology | standard | guideline",
      "name": "человеко-понятное название документа",
      "reason": "зачем этот документ нужен для ответа на данное письмо",
      "priority": "high | medium | low"
    }}
  ],
  "recommended_docs": [
    {{
      "code": "string, короткий код (например, 'TPL_COMPLAINT_REFUND', 'CASE_SIMILAR_COMPLAINT')",
      "kind": "template | case | faq | playbook | example",
      "name": "человеко-понятное название/тип материала",
      "reason": "как это поможет сформировать ответ",
      "priority": "high | medium | low"
    }}
  ]
}}

Никакого другого текста, комментариев или Markdown добавлять нельзя.

--------------------------------
ВХОДНЫЕ ДАННЫЕ О ПИСЬМЕ
--------------------------------

=== ТЕКСТ ПИСЬМА ===
{letter_text}

=== РЕЗУЛЬТАТ NER (JSON) ===
{ner_json}

=== РЕЗУЛЬТАТ КЛАССИФИКАТОРА (JSON) ===
{classifier_json}

=== РЕЗЮМЕ ОТ LLM1 (JSON) ===
{llm1_json}

Пояснение по полям:
- NER может содержать:
  - contract_numbers (номера договоров),
  - deadlines (сроки и дедлайны),
  - law_refs (ссылки на НПА),
  - contacts, organizations.
- Классификатор:
  - type (тип письма: жалоба, регуляторный запрос, уведомление и т.п.),
  - urgency (срочность),
  - formality,
  - departments (какие подразделения вовлечены),
  - legal_risk (есть/нет юридический риск).
- LLM1:
  - core_request (суть запроса),
  - requirements (что конкретно требует автор),
  - expectations (чего он ожидает).

ЕЩЁ РАЗ: верни ТОЛЬКО JSON строго указанной структуры и ничего больше.
""".strip()

# ------------------------
# Вызов Яндекс LLM
# ------------------------
response = client.responses.create(
    model=MODEL,
    input=prompt,
    max_output_tokens=600,
    temperature=0.0
)

raw = response.output_text.strip()

# ------------------------
# Чистим ```json обёртки, если есть
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
target_keys = {"valid_docs", "recommended_docs"}

valid_objects = []
for js in candidates:
    try:
        obj = json.loads(js)
        valid_objects.append(obj)
    except Exception:
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
        "valid_docs": [],
        "recommended_docs": []
    }

best_obj.setdefault("valid_docs", [])
best_obj.setdefault("recommended_docs", [])

# ------------------------
# Сохраняем в JSON файл
# ------------------------
with open("rag_docs_output.json", "w", encoding="utf-8") as f:
    json.dump(best_obj, f, ensure_ascii=False, indent=2)

print("Файл сохранён → rag_docs_output.json")
