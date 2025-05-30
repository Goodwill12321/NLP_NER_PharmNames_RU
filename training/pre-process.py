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

        # Define the regex to capture JSON objects.
        # It looks for a '{' followed by any characters (non-greedy) and ends with a '}'.
        # re.DOTALL ensures '.' matches newlines.
        json_pattern = re.compile(r'(\{.*?\})', re.DOTALL) # Added parentheses around the whole pattern to capture it

        last_idx = 0
        
        # We need to find all matches and correctly manage the remainder.
        # This requires iterating through the content and finding non-overlapping JSON blocks.
        # A simpler way is to find all JSON-like blocks, process them, and then
        # reconstruct the remainder by removing the successfully processed blocks.

        # Let's find all potential JSON blocks first
        potential_json_strings = []
        for match in json_pattern.finditer(content):
            potential_json_strings.append((match.group(1), match.start(), match.end())) # Capture the group

        processed_indices = [] # To keep track of parts of content that were valid JSON

        for json_str, start_idx, end_idx in potential_json_strings:
            # Clean the string - remove any leading/trailing whitespace or potentially a single quote
            # This handles cases where the original data might have something like: '{...}'
            cleaned_json_str = json_str.strip()
            if cleaned_json_str.startswith("'") and cleaned_json_str.endswith("'"):
                cleaned_json_str = cleaned_json_str[1:-1]
            
            try:
                json_object = json.loads(cleaned_json_str)

                # Check for "ИсходныеДанные" and its type
                if "ИсходныеДанные" in json_object and isinstance(json_object["ИсходныеДанные"], dict):
                    extracted_json_objects.append(json_object)
                    processed_indices.append((start_idx, end_idx)) # Mark this range as processed
                else:
                    error_entries.append({
                        "error_text": "JSON-объект найден, но отсутствует или некорректен 'ИсходныеДанные' (ожидается dict).",
                        "original_data": json_str # Use the original string for error reporting
                    })
            except json.JSONDecodeError as e:
                error_entries.append({
                    "error_text": f"Строка похожа на JSON, но не валидна: {e}",
                    "original_data": json_str # Use the original string for error reporting
                })
            except Exception as e:
                error_entries.append({
                    "error_text": f"Непредвиденная ошибка при обработке JSON: {e}",
                    "original_data": json_str # Use the original string for error reporting
                })
        
        # Now reconstruct the remainder parts
        current_idx = 0
        processed_indices.sort() # Ensure indices are sorted to process remainder correctly

        for start, end in processed_indices:
            if start > current_idx:
                remainder_parts.append(content[current_idx:start].strip())
            current_idx = end
        
        if current_idx < len(content):
            remainder_parts.append(content[current_idx:].strip())

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
        cleaned_remainder = "\n".join(part for part in remainder_parts if part) 
        if cleaned_remainder.strip():
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