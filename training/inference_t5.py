import json
import re
import csv
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
from  train_t5 import to_camel_case, normalize_keys_to_camel_case, target_fields

# === НАСТРОЙКИ ===
MODEL_PATH = "./t5-med-ner"
INPUT_FILE = "./data/examples_many.csv"  # или .csv
OUTPUT_FILE = "./data/predictions.jsonl"



tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

# === УТИЛИТЫ ===
hints_fields = [ "ТорговоеНаименование",
      "Дозировка",
      "ЛекФорма",
      "ПервичнаяУпаковкаНазвание",
      "ПервичнаяУпаковкаКоличество",
      "ПотребительскаяУпаковкаКолво",
      "ВторичнаяУпаковкаНазвание",
      "ВторичнаяУпаковкаКоличество"]

def normalize_keys_to_lower(d: dict) -> dict:
    return {k.lower(): v for k, v in d.items()}

def build_input_text(entry):
    product = entry.get("Товар_Поставки", entry.get("Товарпоставки", ""))
    entry_lower = normalize_keys_to_lower(entry)
    hints = []
    for field in hints_fields:
        value = entry_lower.get(field.lower(), "")
        if value not in [None, ""]:
            hints.append(f"{field}: {value}")

    return "\n".join([
        "Задание: Извлеки части наименования из товара.",
        f"Product: {product}",
        "Hints:",
        *hints
    ])

def predict(entry):
    input_text = build_input_text(entry)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    start_time = time.time()
    outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    end_time = time.time()
    print(f"⏱️ Время генерации: {end_time - start_time:.2f} секунд")
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"🔍 decoded: {decoded}")
    return decoded.replace("\n", " ").strip()
    """ try:
        result = json.loads(decoded)
    except json.JSONDecodeError:
        result = {}
    return result """
    #return {field: result.get(field, None) for field in target_fields}

# === ЗАГРУЗКА ДАННЫХ ===
def read_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def read_csv(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

def main():
    input_path = Path(INPUT_FILE)
    if input_path.suffix == ".jsonl":
        examples = list(read_jsonl(input_path))
    elif input_path.suffix == ".csv":
        examples = list(read_csv(input_path))
    else:
        raise ValueError("Поддерживаются только .jsonl и .csv")

    # === ПРЕДСКАЗАНИЕ ===
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for i, raw_entry in enumerate(examples):
            entry = normalize_keys_to_camel_case(raw_entry)
            prediction = predict(entry)

            result = {
                "гуид_записи": entry.get("гуид_записи"),
                "товар_поставки": entry.get("товар_поставки", entry.get("Товарпоставки")),
                "prediction": prediction
            }

            out_f.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(f"[{i+1}] ✅ обработано")

if __name__ == "__main__":
    main()