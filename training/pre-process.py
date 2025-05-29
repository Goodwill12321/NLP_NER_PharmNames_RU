import re
import json

def extract_and_save_json_with_validation_and_errors(input_filepath, output_filepath, error_filepath, remainder_filepath):
    """
    Извлекает JSON-объекты из текстового файла, разделенные произвольными разделителями.
    Проверяет наличие вложенного элемента "ИсходныеДанные".
    Сохраняет валидные JSON-объекты в output_filepath как массив.
    Ошибочные элементы сохраняет в error_filepath с указанием ошибки.
    Оставшийся текст (не JSON) сохраняет в remainder_filepath.
    """
    extracted_json_objects = []
    error_entries = []
    remainder_parts = []

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Регулярное выражение для поиска JSON-объектов.
        # Используем нежадный поиск с re.DOTALL.
        # Более сложное regex для проверки вложенности JSON может быть очень сложным
        # и часто менее надежным, чем последующая валидация через json.loads().
        # Поэтому, мы сначала ищем любые блоки {...}, а затем проверяем их содержимое.
        json_pattern = re.compile(r'\{.*?\}', re.DOTALL)

        last_idx = 0
        for match in json_pattern.finditer(content):
            # Сохраняем текст до текущего совпадения
            if match.start() > last_idx:
                remainder_parts.append(content[last_idx:match.start()].strip())

            json_string = match.group(0)
            try:
                json_object = json.loads(json_string)

                # Проверка на наличие "ИсходныеДанные" и его тип
                if "ИсходныеДанные" in json_object and isinstance(json_object["ИсходныеДанные"], dict):
                    extracted_json_objects.append(json_object)
                else:
                    error_entries.append({
                        "error_text": "JSON-объект найден, но отсутствует или некорректен 'ИсходныеДанные' (ожидается dict).",
                        "original_data": json_string
                    })
            except json.JSONDecodeError as e:
                error_entries.append({
                    "error_text": f"Строка похожа на JSON, но не валидна: {e}",
                    "original_data": json_string
                })
            except Exception as e:
                error_entries.append({
                    "error_text": f"Непредвиденная ошибка при обработке JSON: {e}",
                    "original_data": json_string
                })
            
            last_idx = match.end()
        
        # Сохраняем оставшийся текст после последнего совпадения
        if last_idx < len(content):
            remainder_parts.append(content[last_idx:].strip())


        # --- Запись результатов ---

        # 1. Запись валидных JSON-объектов
        if extracted_json_objects:
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                json.dump(extracted_json_objects, outfile, ensure_ascii=False, indent=2)
            print(f"Успешно извлечено {len(extracted_json_objects)} валидных JSON-объектов и сохранено в {output_filepath}")
        else:
            print(f"Не найдено ни одного валидного JSON-объекта с 'ИсходныеДанные' в файле {input_filepath}.")

        # 2. Запись ошибок
        if error_entries:
            with open(error_filepath, 'w', encoding='utf-8') as errfile:
                for entry in error_entries:
                    errfile.write(f"--- Ошибка: {entry['error_text']} ---\n")
                    errfile.write(entry['original_data'])
                    errfile.write("\n/***********ОКОНЧАНИЕ ЭЛЕМЕНТА С ОШИБКОЙ***********/\n\n")
            print(f"Обнаружено {len(error_entries)} ошибочных элементов. Подробности в {error_filepath}")
        else:
            print("Ошибочных элементов не найдено.")

        # 3. Запись остатка
        cleaned_remainder = "\n".join(part for part in remainder_parts if part) # Удаляем пустые строки
        if cleaned_remainder.strip(): # Проверяем, что остаток не пустой после strip
            with open(remainder_filepath, 'w', encoding='utf-8') as remfile:
                remfile.write(cleaned_remainder)
            print(f"Оставшийся текст сохранен в {remainder_filepath}")
        else:
            print("Оставшегося текста (не JSON) не обнаружено.")

    except FileNotFoundError:
        print(f"Ошибка: Файл '{input_filepath}' не найден.")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")

# --- Использование скрипта ---


input_file = './data/or_annotated_deepseekr_4.jsonl'
output_file = 'output.json'
error_file = 'errors.txt'
remainder_file = 'remainder.txt'

extract_and_save_json_with_validation_and_errors(input_file, output_file, error_file, remainder_file)