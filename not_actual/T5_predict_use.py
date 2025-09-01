import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import torch
import re
import os
import random

# === Параметры ===
excel_path = "./data/nlp_dataset_distinct.xlsx"
output_path = "./data/predicted_tokenized_output_random_sample.xlsx"
model_path = "./flan_t5_medsplit/checkpoint-final"
sheet_name = 0

start_row = 100       # Стартовая строка (включительно)
end_row = 1000        # Конечная строка (исключительно)
sample_size = 10      # Сколько случайных строк выбрать

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Проверка и загрузка модели ===
assert os.path.isdir(model_path), f"❌ Не найдена директория модели: {model_path}"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
model.eval()

# === Подготовка входного текста ===
def build_input(example: dict) -> str:
    fields = [
        "ТоварПоставки", "ПредставлениеТовара", "ТорговоеНаименование",
        "Дозировка", "ЛекФорма", "ПервичнаяУпаковкаНазвание",
        "ПервичнаяУпаковкаКоличество", "ВторичнаяУпаковкаНазвание",
        "ВторичнаяУпаковкаКоличество", "ПотребительскаяУпаковкаКолво"
    ]
    return " | ".join([f"{field}: {example.get(field, '')}" for field in fields])

# === Предсказание и разбор ===
def predict_parts(example: dict) -> dict:
    input_text = build_input(example)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("🔎 decoded:", decoded)
    result = {}
    for match in re.finditer(r"([\w_]+):\s?([^|]+)", decoded):
        key, value = match.group(1).strip(), match.group(2).strip()
        result[f"{key}_ТП_pred"] = value
    return result

# === Чтение диапазона строк и случайная выборка ===
print(f"\n🔄 Чтение строк {start_row}–{end_row} из Excel: {excel_path}")
df_range = pd.read_excel(
    excel_path,
    sheet_name=sheet_name,
    skiprows=range(1, start_row + 1),
    nrows=end_row - start_row
).fillna("")

# === Случайная выборка N строк ===
assert sample_size <= len(df_range), "❌ sample_size больше количества доступных строк в диапазоне!"
df_sample = df_range.sample(n=sample_size, random_state=42).reset_index(drop=True)

# === Предсказания ===
tqdm.pandas(desc="🤖 Предсказание токенов")
predictions = df_sample.progress_apply(lambda row: predict_parts(row.to_dict()), axis=1)

# === Объединение с оригиналом ===
predicted_df = pd.DataFrame(predictions.tolist())
result_df = pd.concat([df_sample, predicted_df], axis=1)

# === Сохранение ===
result_df.to_excel(output_path, index=False)
print(f"\n✅ {sample_size} случайных строк предсказаны и сохранены в: {output_path}")
