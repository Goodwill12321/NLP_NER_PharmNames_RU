import re
import json

def extract_and_save_json_with_validation_and_errors(input_filepath, output_filepath, error_filepath, remainder_filepath):
    """
    Извлекает JSON-объекты из текстового файла, разделенные произвольными разделителями.
    Пытается "достроить" JSON-объект, ища следующую '}' или добавляя её искусственно.
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

        # Регулярное выражение для поиска блоков, начинающихся с '{' и заканчивающихся '}'
        # Нежадный поиск (.*?) предотвращает захват слишком большого куска текста.
        # Мы ищем первую закрывающую скобку. Дальнейшая логика будет искать вторую, если нужно.
        json_pattern = re.compile(r'(\{.*?\})', re.DOTALL)

        last_idx = 0
        
        # Для корректного управления остатком, мы будем итерироваться по содержимому
        # ища JSON-блоки и отмечая обработанные части.
        
        # Используем finditer для получения всех совпадений и их позиций
        # Но обработку будем делать последовательно, чтобы правильно управлять last_idx
        
        current_pos = 0 # Текущая позиция в тексте, откуда начинаем поиск
        
        while current_pos < len(content):
            match = json_pattern.search(content, current_pos)
            
            if not match:
                # Если больше нет JSON-подобных блоков, добавляем остаток и выходим
                if current_pos < len(content):
                    remainder_parts.append(content[current_pos:].strip())
                break # Выход из цикла while

            # Сохраняем текст до текущего совпадения как часть остатка
            if match.start() > current_pos:
                remainder_parts.append(content[current_pos:match.start()].strip())

            json_string_initial = match.group(0) # Исходная строка, найденная regex
            current_json_end_idx = match.end() # Конец текущего захваченного regex'ом JSON-блока

            final_json_string = json_string_initial
            potential_second_brace_found = False
            
            # Ищем вторую закрывающую скобку, пропуская только пробельные символы
            temp_idx = current_json_end_idx
            
            while temp_idx < len(content) and content[temp_idx].isspace():
                temp_idx += 1
            
            if temp_idx < len(content) and content[temp_idx] == '}':
                # Найдена вторая закрывающая скобка, добавляем её к JSON-строке
                final_json_string = json_string_initial + content[current_json_end_idx : temp_idx + 1]
                current_json_end_idx = temp_idx + 1 # Обновляем конец захваченного блока
                potential_second_brace_found = True

            # Предварительная очистка: убираем внешние одинарные кавычки, если есть
            cleaned_json_str = final_json_string.strip()
            if cleaned_json_str.startswith("'") and cleaned_json_str.endswith("'"):
                cleaned_json_str = cleaned_json_str[1:-1]

            cleaned_json_str = cleaned_json_str.replace("Исходные_данные", "ИсходныеДанные")
            cleaned_json_str = cleaned_json_str.replace("Колво", "Количество")

            parsed_successfully = False
            error_reason = ""

            try:
                json_object = json.loads(cleaned_json_str)
                parsed_successfully = True
            except json.JSONDecodeError as e:
                error_reason = str(e)
                # Если ошибка связана с "нехваткой" закрывающей скобки
                # и мы ещё не добавляли её
                if not potential_second_brace_found and \
                   ("Expecting '}'" in error_reason or "Unterminated string" in error_reason or "Unexpected EOF" in error_reason):
                    
                    # Попытка 2: добавляем закрывающую скобку и пробуем снова
                    try:
                        json_object = json.loads(cleaned_json_str + '}')
                        parsed_successfully = True
                        print(f"Исправлен JSON: добавлена искусственная закрывающая скобка для объекта, начинающегося с: {cleaned_json_str[:50]}...")
                       
                    except json.JSONDecodeError as e2:
                        error_entries.append({
                            "error_text": f"JSON-строка не валидна даже после добавления '}}: {e2}",
                            "original_data": final_json_string # Исходная строка до добавления
                        })
                else:
                    error_entries.append({
                        "error_text": f"JSON-строка не валидна: {error_reason}",
                        "original_data": final_json_string # Исходная строка
                    })
            except Exception as e:
                error_entries.append({
                    "error_text": f"Непредвиденная ошибка при обработке JSON: {e}",
                    "original_data": final_json_string
                })
            
            # Если объект успешно распарсен (после 1 или 2 попытки)
            if parsed_successfully:
                # Проверка на наличие "ИсходныеДанные" и его тип
                if "ИсходныеДанные" in json_object and isinstance(json_object["ИсходныеДанные"], dict):
                    extracted_json_objects.append(json_object)
                    # Обновляем позицию для следующего поиска
                    current_pos = current_json_end_idx
                else:
                    error_entries.append({
                        "error_text": "JSON-объект найден, но отсутствует или некорректен 'ИсходныеДанные' (ожидается dict).",
                        "original_data": final_json_string
                    })
                    current_pos = current_json_end_idx # Двигаем указатель, чтобы не обрабатывать этот же блок снова
            else:
                # Если парсинг не удался ни после одной, ни после двух попыток,
                # мы все равно должны продвинуть указатель, чтобы избежать бесконечного цикла.
                # Продвигаем его до конца текущего захваченного regex'ом блока + 1,
                # или до следующей открывающей скобки, чтобы избежать зацикливания на поврежденных данных.
                current_pos = current_json_end_idx
                
                # Более продвинутый сдвиг: ищем следующую '{'
                next_open_brace = content.find('{', current_pos)
                if next_open_brace != -1:
                    current_pos = next_open_brace
                else:
                    current_pos = len(content) # Если больше нет '{', переходим в конец файла
                

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

# Создайте тестовый файл для проверки нового поведения
# Например, 'input_test_final.txt'


input_file = './data/or_annotated_deepseekr.jsonl' # Укажите ваш файл
output_file = './data/train_nlp.json'
error_file = './data/pre/errors.txt'
remainder_file = './data/pre/remainder.txt'

extract_and_save_json_with_validation_and_errors(input_file, output_file, error_file, remainder_file)