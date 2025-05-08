
from openpyxl import load_workbook
from tqdm import tqdm
import json

# === Настройки ===
input_file = "nlp_dataset.xlsx"
output_file = "for_labelstudio_streamed.jsonl"
excluded_columns = []

def make_hint(row_dict):
    return "; ".join(
        f"{k}: {v}" for k, v in row_dict.items()
        if k not in excluded_columns and v
    )

def main():
    print("📥 Открытие Excel-файла...")
    wb = load_workbook(filename=input_file, read_only=True)
    ws = wb.active  # используем первый лист

    # Получаем заголовки
    headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]

    # Подсчёт количества строк для tqdm
    total_rows = ws.max_row - 1  # без заголовка

    print(f"✅ Всего строк: {total_rows}")

    with open(output_file, "w", encoding="utf-8") as f:
        for row in tqdm(ws.iter_rows(min_row=2), total=total_rows, desc="Обработка строк"):
            row_dict = {headers[i]: (str(cell.value).strip() if cell.value is not None else "") for i, cell in enumerate(row)}
            item = {
                "text": row_dict.get("ТоварПоставки", ""),
                "hints": make_hint(row_dict)
            }
            # Добавляем все остальные колонки
            for key, val in row_dict.items():
                if key not in item:
                    item[key] = val
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ Готово: {output_file}")

if __name__ == "__main__":
    main()
