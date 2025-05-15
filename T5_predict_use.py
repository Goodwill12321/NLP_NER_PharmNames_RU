import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import torch
import re

# === Параметры ===
excel_path = "./data/nlp_dataset_distinct.xlsx"         # Входной файл
output_path = "./data/predicted_tokenized_output_split.xlsx"
model_path = "./flan_t5_medsplit/checkpoint-final"                     # Путь к модели
sheet_name = 0

start_row = 100      # С какой строки начать
end_row = 110      # По какую строку (исключительно)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Загрузка модели ===
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
model.eval()

# === Подготовка текста для подачи в модель ===
def build_input(example: dict) -> str:
    parts = [
        f"ТоварПоставки: {example.get('ТоварПоставки', '')}",
        f"ПредставлениеТовара: {example.get('ПредставлениеТовара', '')}",
        f"ТорговоеНаименование: {example.get('ТорговоеНаименование', '')}",
        f"Дозировка: {example.get('Дозировка', '')}",
        f"ЛекФорма: {example.get('ЛекФорма', '')}",
        f"ПервичнаяУпаковкаНазвание: {example.get('ПервичнаяУпаковкаНазвание', '')}",
        f"ПервичнаяУпаковкаКоличество: {example.get('ПервичнаяУпаковкаКоличество', '')}",
        f"ВторичнаяУпаковкаНазвание: {example.get('ВторичнаяУпаковкаНазвание', '')}",
        f"ВторичнаяУпаковкаКоличество: {example.get('ВторичнаяУпаковкаКоличество', '')}",
        f"ПотребительскаяУпаковкаКолво: {example.get('ПотребительскаяУпаковкаКолво', '')}",
        
    ]
    return " | ".join(parts)

# === Предсказание и разбор на поля ===
def predict_parts(example: dict) -> dict:
    input_text = build_input(example)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_length=256)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Разбор строки на ключи и добавление постфикса
    result = {}
    for match in re.finditer(r"(\w+):\s?([^|]+)", decoded):
        key, value = match.group(1).strip(), match.group(2).strip()
        result[f"{key}_ТП_pred"] = value
    return result

# === Стриминг чтения Excel и обработка только нужных строк ===
print(f"\n🔄 Чтение строк {start_row}–{end_row} из Excel...")
df_chunk = pd.read_excel(
    excel_path,
    sheet_name=sheet_name,
    skiprows=range(1, start_row + 1),  # Пропускаем заголовок + предыдущие строки
    nrows=end_row - start_row
)
df_chunk = df_chunk.fillna("")

# === Прогноз ===
tqdm.pandas(desc="🤖 Предсказание")
predictions = df_chunk.progress_apply(lambda row: predict_parts(row.to_dict()), axis=1)

# === Объединение с оригинальными данными ===
predicted_df = pd.DataFrame(predictions.tolist())
result_df = pd.concat([df_chunk.reset_index(drop=True), predicted_df], axis=1)

# === Сохранение результата ===
result_df.to_excel(output_path, index=False)
print(f"\n✅ Предсказания сохранены в: {output_path}")
