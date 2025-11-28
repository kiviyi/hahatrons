# pipeline.py
import json
import subprocess


def run_script(script_name):
    """Запускает Python-скрипт как отдельный процесс."""
    print(f"Запуск: {script_name}")
    subprocess.run(["python", script_name], check=True)


def pipeline(letter_text: str):
    """Полный пайплайн: письмо → NER → Classifier → LLM1 → LLM2 → LLM3."""

    # ---------------------------------------------------------
    # 1. Сохраняем письмо
    # ---------------------------------------------------------
    with open("letter.txt", "w", encoding="utf-8") as f:
        f.write(letter_text.strip())

    print("Письмо сохранено в letter.txt")

    # ---------------------------------------------------------
    # 2. Запуск NER
    # ---------------------------------------------------------
    run_script("ner_yandex.py")

    with open("ner_output.json", "r", encoding="utf-8") as f:
        ner_result = json.load(f)

    # ---------------------------------------------------------
    # 3. Запуск классификатора
    # ---------------------------------------------------------
    run_script("app_final_yandex.py")

    with open("classification_output.json", "r", encoding="utf-8") as f:
        classifier_result = json.load(f)

    # ---------------------------------------------------------
    # 4. Запуск LLM1 (извлечение сути)
    # ---------------------------------------------------------
    run_script("llm1_yandex.py")

    with open("llm1_output.json", "r", encoding="utf-8") as f:
        llm1_result = json.load(f)

    # ---------------------------------------------------------
    # 5. Запуск LLM2 (генерация 4 ответов)
    # ---------------------------------------------------------
    # В llm2_yandex.py данные читаются из файлов,
    # поэтому здесь никаких аргументов передавать не нужно.
    run_script("llm2_yandex.py")

    with open("llm2_output.json", "r", encoding="utf-8") as f:
        llm2_result = json.load(f)

    # ---------------------------------------------------------
    # 6. Запуск LLM3 (комплаенс-проверка)
    # ---------------------------------------------------------
    run_script("llm3_yandex.py")

    with open("llm3_output.json", "r", encoding="utf-8") as f:
        llm3_result = json.load(f)

    print("\nПайплайн завершён!")
    print("Финальный результат находится в llm3_output.json")

    return llm3_result


# ---------------------------------------------------------
# Пример использования
# ---------------------------------------------------------
if __name__ == "__main__":

    with open('tt.txt', 'r', encoding='utf-8') as file:
        test_letter = file.read()
    # Теперь переменная content содержит весь текст из файла
    print(test_letter)

    result = pipeline(test_letter)
    print("\n=== Финальный ответ LLM3 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))