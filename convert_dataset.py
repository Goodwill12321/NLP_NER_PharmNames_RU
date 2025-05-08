from openpyxl import load_workbook
from tqdm import tqdm
import json

input_file = "nlp_dataset.xlsx"
output_file = "for_labelstudio_streamed_fixed.jsonl"

# ❗ Укажи здесь, какие колонки НЕ нужно размечать (но можно показывать в hints)
excluded_columns = [
    "ТоварПоставки", "ПредставлениеТовара", "МНН", "МННСтрокой",
    "ЖН", "СМНН", "КЛП", "Наркотический", "ОКПД2",
    "РегНомерПредЦены", "ДатаОкончанияПредЦены",
    "ВидЗаписиРеестраЖН", "НомерРешенияРеестраЖН",
    "ДатаРегистрацииЦеныРеестраЖН", "ДатаВступленияВСилуРеестраЖН",
    "ДозировкаГРЛС", "Дозировка_ЕИ_ПотребительскаяОКЕИ", "Штрихкод", 
    "ВладелецНаименование", "ВладелецСтрана", "ЛекформаДозировкаУпаковкаИзРеестраЖН", "ВладелецРУИзРеестраЖН",
    "НомерРешенияРеестраЖН", "НомерРУ", "ДатаРУ"
]

def make_hint(row_dict):
    return "; ".join(
        f"{k}: {v}" for k, v in row_dict.items()
        if v and k not in excluded_columns
    )


def main():
    print("📥 Открытие Excel-файла...")
    wb = load_workbook(filename=input_file, read_only=True)
    ws = wb.active

    headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    total_rows = ws.max_row - 1

    print(f"✅ Всего строк: {total_rows}")
    print(f"❗ Колонки, исключённые из разметки: {excluded_columns}")

    with open(output_file, "w", encoding="utf-8") as f:
        for row in tqdm(ws.iter_rows(min_row=2), total=total_rows, desc="Обработка строк"):
            row_dict = {headers[i]: (str(cell.value).strip() if cell.value is not None else "") for i, cell in enumerate(row)}
            # data включает все колонки
            data_item = {key: val for key, val in row_dict.items()}
            data_item["hints"] = make_hint(row_dict)
            json_line = {"data": data_item}
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"\n✅ Готово для импорта в Label Studio: {output_file}")

if __name__ == "__main__":
    main()
