from openpyxl import load_workbook
import json
from tqdm import tqdm

def convert_to_labelstudio_json(input_file, output_file):
    EXCLUDED_COLUMNS = {
        "ТоварПоставки", "ПредставлениеТовара", "МНН", "МННСтрокой",
        "ЖН", "СМНН", "КЛП", "Наркотический", "ОКПД2",
        "РегНомерПредЦены", "ДатаОкончанияПредЦены",
        "ВидЗаписиРеестраЖН", "НомерРешенияРеестраЖН",
        "ДатаРегистрацииЦеныРеестраЖН", "ДатаВступленияВСилуРеестраЖН",
        "Дозировка_ЕИ_ПотребительскаяОКЕИ", "Штрихкод", 
        "ВладелецНаименование", "ВладелецСтрана", "ЛекформаДозировкаУпаковкаИзРеестраЖН", 
        "ВладелецРУИзРеестраЖН", "НомерРешенияРеестраЖН", "НомерРУ", "ДатаРУ"
    }

    print("🔍 Чтение Excel файла...")
    wb = load_workbook(filename=input_file, read_only=True)
    ws = wb.active
    
    headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    total_rows = ws.max_row - 1
    
    # Собираем все данные в список
    tasks = []
    
    for row in tqdm(ws.iter_rows(min_row=2), total=total_rows, desc="Конвертация"):
        row_data = {header: str(cell.value) if cell.value is not None else "" 
                  for header, cell in zip(headers, row)}
        
        tasks.append({
            "data": {
                "text": row_data.get("ТоварПоставки", ""),
                "meta": {
                    "reference": row_data.get("ПредставлениеТовара", ""),
                    **{k: v for k, v in row_data.items() 
                       if k not in EXCLUDED_COLUMNS}
                }
            }
        })
    
    # Записываем как единый JSON-файл
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump({"tasks": tasks}, f_out, ensure_ascii=False, indent=2)
    
    wb.close()
    print(f"✅ Результат сохранён в {output_file} (формат JSON)")

convert_to_labelstudio_json("nlp_dataset.xlsx", "output.json")