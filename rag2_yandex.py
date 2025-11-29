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

# Исходное письмо
with open("letter.txt", "r", encoding="utf-8") as f:
    letter_text = f.read().strip()

# Документы, подобранные RAG1
with open("rag_docs_output.json", "r", encoding="utf-8") as f:
    rag_docs = json.load(f)

# Черновики ответов LLM2
with open("llm2_output.json", "r", encoding="utf-8") as f:
    llm2_result = json.load(f)

rag_docs_json = json.dumps(rag_docs, ensure_ascii=False, indent=2)
llm2_json     = json.dumps(llm2_result, ensure_ascii=False, indent=2)

# ------------------------
# PROMPT: анализ использования документов
# ------------------------
prompt = f"""
Ты — юридический и комплаенс-эксперт крупного банка.

Тебе даются:
1) Исходное письмо клиента.
2) Список документов, которые система RAG рекомендовала использовать (valid_docs и recommended_docs).
3) Черновики ответов Банка (4 варианта от LLM2).

ТВОЯ ЗАДАЧА:
Для каждого варианта ответа (official, business, client_friendly, simple):
- Определить, на какие документы из списка он явно или неявно опирается (по смыслу, по формулировкам).
- Определить, какие важные документы из списка были проигнорированы (их логично было бы учесть, но содержание ответа этого не отражает).
- Найти упоминания нормативных актов, регламентов или внутренних документов, которых НЕТ в списке документов (потенциальные галлюцинации).
- Дать краткий комментарий по юридической аккуратности с точки зрения использования документов.

ОГРАНИЧЕНИЯ:
- НЕЛЬЗЯ придумывать новые документы, которых нет ни в ответах, ни в списках RAG; можно лишь отмечать, что ответ в целом ссылается на "нормативную базу" без указания конкретного документа.
- Если ответ вообще не опирается на документы, так и напиши.
- Если ответ содержит формулировки вроде «в соответствии с действующим законодательством», но без ссылки на конкретный акт, можешь упомянуть это только в комментарии, не добавляя несуществующие документы в used_docs.

ФОРМАТ ОТВЕТА:
Верни ОДИН валидный JSON следующего вида:

{{
  "analysis": {{
    "official": {{
      "used_docs": ["КОД_1", "КОД_2"],
      "missing_docs": ["КОД_3"],
      "hallucinated_refs": ["описание проблемы 1", "описание проблемы 2"],
      "comment": "краткий вывод по этому варианту"
    }},
    "business": {{
      "used_docs": [...],
      "missing_docs": [...],
      "hallucinated_refs": [...],
      "comment": "..."
    }},
    "client_friendly": {{
      "used_docs": [...],
      "missing_docs": [...],
      "hallucinated_refs": [...],
      "comment": "..."
    }},
    "simple": {{
      "used_docs": [...],
      "missing_docs": [...],
      "hallucinated_refs": [...],
      "comment": "..."
    }}
  }}
}}

Где:
- "used_docs" — список code из входного списка документов (valid_docs/recommended_docs), которые фактически используются или явно подразумеваются в этом варианте ответа.
- "missing_docs" — список code документов, которые было бы логично учесть для такого ответа, но в тексте ответа нет никакого отражения их содержания.
- "hallucinated_refs" — список текстовых комментариев о ссылках на документы/нормативные акты, которых НЕТ в списке документов.
- "comment" — короткий текстовый вывод по используемым документам и рискам.

Никакого другого текста, комментариев или Markdown добавлять нельзя.

--------------------------------
ИСХОДНОЕ ПИСЬМО
--------------------------------
{letter_text}

--------------------------------
СПИСОК ДОКУМЕНТОВ ОТ RAG1 (JSON)
--------------------------------
{rag_docs_json}

--------------------------------
ЧЕРНОВИКИ ОТВЕТОВ LLM2 (JSON)
--------------------------------
{llm2_json}
""".strip()

# ------------------------
# Вызов Яндекс LLM
# ------------------------
response = client.responses.create(
    model=MODEL,
    input=prompt,
    max_output_tokens=800,
    temperature=0.0
)

raw = response.output_text.strip()

# ------------------------
# Чистим ```json обёртки
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
target_keys = {"analysis"}

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
        "analysis": {
            "official": {
                "used_docs": [],
                "missing_docs": [],
                "hallucinated_refs": [],
                "comment": "анализ не удалось выполнить"
            },
            "business": {
                "used_docs": [],
                "missing_docs": [],
                "hallucinated_refs": [],
                "comment": "анализ не удалось выполнить"
            },
            "client_friendly": {
                "used_docs": [],
                "missing_docs": [],
                "hallucinated_refs": [],
                "comment": "анализ не удалось выполнить"
            },
            "simple": {
                "used_docs": [],
                "missing_docs": [],
                "hallucinated_refs": [],
                "comment": "анализ не удалось выполнить"
            }
        }
    }

analysis = best_obj.setdefault("analysis", {})
for key in ["official", "business", "client_friendly", "simple"]:
    analysis.setdefault(key, {})
    analysis[key].setdefault("used_docs", [])
    analysis[key].setdefault("missing_docs", [])
    analysis[key].setdefault("hallucinated_refs", [])
    analysis[key].setdefault("comment", "")

# ------------------------
# Сохраняем в JSON файл
# ------------------------
with open("rag_usage_output.json", "w", encoding="utf-8") as f:
    json.dump(best_obj, f, ensure_ascii=False, indent=2)

print("Файл сохранён → rag_usage_output.json")
