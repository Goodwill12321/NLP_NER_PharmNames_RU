import pandas as pd
import json
from tqdm import tqdm

# Путь к Excel-файлу
excel_path = "nlp_dataset_distinct.xlsx"
output_json_path = "train_flan_t5.json"

# Колонки, которые используются
source_columns = [
    "ПредставлениеТовара",
    "ТорговоеНаименование",
    "ПроизводительНаименование",
    "ПроизводительСтрана",
    "Дозировка",
    "ЛекФорма",
    "ПервичнаяУпаковкаНазвание",
    "ПервичнаяУпаковкаКоличество",
    "Дозировка_ЕИ",
    "Дозировка_ЕИ_Потребительская",
    "ДозировкаКоличество",
    "ПотребительскаяУпаковкаКолво",
    "ВторичнаяУпаковкаНазвание",
    "ВторичнаяУпаковкаКоличество",
]

target_columns = [
    "ТорговоеНаименование_ТП",
    "Производитель_ТП",
    "Страна_ТП",
    "Дозировка_ТП",
    "Лекформа_ТП",
    "ПервичнаяУпаковкаНазвание_ТП",
    "ПервичнаяУпаковкаКоличество_ТП",
    "Дозировка_ЕИ_ТП",
    "Дозировка_ЕИ_Потребительская_ТП",
    "ДозировкаКоличество_ТП",
    "ПотребительскаяУпаковкаКолво_ТП",
    "ВторичнаяУпаковкаНазвание_ТП",
    "ВторичнаяУпаковкаКоличество_ТП",
]

# Подготовка JSON-файла
with open(output_json_path, "w", encoding="utf-8") as f_out:
    f_out.write("[\n")
    first = True
    total = sum(1 for _ in pd.read_excel(excel_path, engine="openpyxl"))
    reader = pd.read_excel(excel_path, chunksize=1, engine="openpyxl")

    for chunk in tqdm(reader, total=total, desc="Обработка строк"):
        row = chunk.iloc[0]
        input_text = f"ТоварПоставки: {row.get('ТоварПоставки', '')}\n"
        input_text += "Эталонные значения:\n"
        input_text += "\n".join(
            f"{col}: {row[col]}" for col in source_columns if pd.notna(row.get(col))
        )

        output_text = "\n".join(
            f"{col.replace('_ТП', '')}: {row[col]}" for col in target_columns if pd.notna(row.get(col))
        )

        if output_text.strip():
            json_obj = {
                "input": input_text.strip(),
                "output": output_text.strip()
            }
            if not first:
                f_out.write(",\n")
            else:
                first = False
            json.dump(json_obj, f_out, ensure_ascii=False, indent=2)

    f_out.write("\n]\n")

print(f"\n✅ Сохранено в {output_json_path}")
