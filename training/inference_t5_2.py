import json
import re
import csv
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
from  train_t5 import to_camel_case, normalize_keys_to_camel_case, target_fields
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === НАСТРОЙКИ ===
MODEL_PATH = "./model/checkpoint-864"
INPUT_FILE = "./data/examples_many.csv"  # или .csv
OUTPUT_FILE = "./data/predictions.jsonl"


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
#tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
#model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

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
    product = entry.get("Товар_Поставки", entry.get("ТоварПоставки", ""))
   # entry_lower = normalize_keys_to_lower(entry)
    """ hints = []
    for field in hints_fields:
        value = entry_lower.get(field.lower(), "")
        if value not in [None, ""]:
            hints.append(f"{field}: {value}") """

    return "\n".join([
       "Задание: Извлеки части из названия лекарственного препарата или товара фармацевтического назначения.",
            f"Наименование товара: {product}",
            "Описание полей:",
            "ТорговоеНаименование_ТП — торговая марка",
            "Дозировка_ТП — количество, концентрация активного вещества (например, мг, мл, мг/мл и др.).",
            "ЛекФорма_ТП — форма выпуска: таблетки, капсулы, мазь и др.",
            "ПервичнаяУпаковкаНазвание_ТП — вид упаковки: блистер, флакон и др.",
            "ПервичнаяУпаковкаКоличество_ТП — количество единиц в первичной уп.",
            "ПотребительскаяУпаковкаКоличество_ТП — количество единиц в потребительской уп..",
            "ВторичнаяУпаковкаНазвание_ТП — название вторичной уп. (например, коробка).",
            "ВторичнаяУпаковкаКоличество_ТП — количество первичных уп. в ней.",
            "Если какие-либо части не определяются, указывай null.",
         
    ])

def predict(entry):
    input_text = build_input_text(entry)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    start_time = time.time()
    outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
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
                "гуид_записи": entry.get("ГУИД_Записи"),
                "товар_поставки": entry.get("товар_поставки", entry.get("ТоварПоставки")),
                "prediction": prediction
            }

            out_f.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(f"[{i+1}] ✅ обработано")

if __name__ == "__main__":
    main()