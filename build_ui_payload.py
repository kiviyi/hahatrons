# build_ui_payload.py
import json
from pathlib import Path

BASE = Path(".")

def load_json(name, default):
    path = BASE / name
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # NER
    ner = load_json("ner_output.json", {
        "contract_numbers": [],
        "deadlines": [],
        "law_refs": [],
        "contacts": {},
        "organizations": []
    })

    # Классификация
    classification = load_json("classification_output.json", {
        "type": None,
        "urgency": None,
        "formality": None,
        "departments": [],
        "legal_risk": None,
        "deadline": None
    })

    # Ответы LLM2
    llm2 = load_json("llm2_output.json", {"answers": {}})
    answers = llm2.get("answers", {})

    # Документы из RAG1
    rag_docs = load_json("rag_docs_output.json", {
        "valid_docs": [],
        "recommended_docs": []
    })

    # Дедлайн для UI: сначала пытаемся взять из classification, если нет — из NER
    ui_deadline = classification.get("deadline")
    if not ui_deadline:
        for d in ner.get("deadlines", []):
            if d.get("type") == "absolute" and d.get("date"):
                ui_deadline = d["date"]
                break

    ui_payload = {
        "ner": ner,
        "classification": {
            "type": classification.get("type"),
            "urgency": classification.get("urgency"),
            "formality": classification.get("formality"),
            "departments": classification.get("departments") or [],
            "legal_risk": classification.get("legal_risk"),
            "deadline": ui_deadline,
        },
        "answers": {
            # маппинг на стили UI
            "formal":   answers.get("official"),
            "business": answers.get("business"),
            "client":   answers.get("client_friendly"),
            "brief":    answers.get("simple"),
        },
        "rag_docs": {
            "valid_docs": rag_docs.get("valid_docs", []),
            "recommended_docs": rag_docs.get("recommended_docs", []),
        },
    }

    with open("ui_payload.json", "w", encoding="utf-8") as f:
        json.dump(ui_payload, f, ensure_ascii=False, indent=2)

    print("Файл сохранён → ui_payload.json")

if __name__ == "__main__":
    main()
