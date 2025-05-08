from openpyxl import load_workbook
import json
from tqdm import tqdm

def convert_excel_to_labelstudio(input_file, output_file):
    # Колонки, которые нужно исключить из меток
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

    print("⏳ Загрузка Excel файла в режиме read_only...")
    try:
        wb = load_workbook(filename=input_file, read_only=True)
        ws = wb.active
        
        # Получаем заголовки
        headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
        total_rows = ws.max_row - 1
        
        print(f"✅ Всего строк: {total_rows}")
        print(f"❗ Колонки, исключённые из разметки: {EXCLUDED_COLUMNS}")

        with open(output_file, "w", encoding="utf-8") as f_out:
            for row in tqdm(ws.iter_rows(min_row=2), total=total_rows, desc="Обработка строк"):
                row_data = {header: cell.value for header, cell in zip(headers, row)}
                
                task_data = {
                    "data": {
                        "text": str(row_data.get("ТоварПоставки", "")),
                        "reference": str(row_data.get("ПредставлениеТовара", "")),
                        "correlated_data": {
                            col: str(row_data.get(col, "")) 
                            for col in EXCLUDED_COLUMNS 
                            if col in row_data
                        },
                        "labeling_data": {
                            col: str(row_data.get(col, "")) 
                            for col in headers 
                            if col not in EXCLUDED_COLUMNS
                        }
                    }
                }
                f_out.write(json.dumps(task_data, ensure_ascii=False) + "\n")
        
        print(f"🎉 Конвертация завершена. Результат сохранён в {output_file}")
        
    except Exception as e:
        print(f"❌ Ошибка при обработке файла: {e}")
    finally:
        wb.close()

if __name__ == "__main__":
    convert_excel_to_labelstudio(
        input_file="nlp_dataset.xlsx",
        output_file="labelstudio_data.jsonl"
    )