import json
import re


def extract_number(value):
    """Извлекает первое число из строки или возвращает само число, если это int/float."""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        match = re.search(r'\d+', value.replace(',', '.'))
        if match:
            return int(match.group(0))
    return None

def contains_whole_number_with_units(text: str, number: int) -> bool:
    """
    Проверяет, входит ли число целиком в строку, не как часть другого числа.
    Допускается, если за числом идёт текст (например, 'мл'), но не другая цифра.
    """
    # Ищем число с возможным префиксом '№', за которым не идёт цифра
    pattern = rf'(?<!\d)(?:№\s*)?{number}(?!\d|{number}\d)'
    return re.search(pattern, text) is not None

def compare_numeric_fields(entry):
    mismatches = []
    str_diff = ''
    for key, value in entry.items():
        if '_ТП' in key and ('Количество' in key or 'Колво' in key):
            base_key = key.replace('_ТП', '')
            base_data = entry.get('ИсходныеДанные', {})
            expected_value = base_data.get(base_key)

            num_tp = extract_number(value)
            if  (value is None or value == '') and contains_whole_number_with_units(entry['ТоварПоставки'], expected_value):
                entry[key] = expected_value
            
            if num_tp is None and base_key != 'ПотребительскаяУпаковкаКоличество':
                continue  
            num_base = extract_number(expected_value)

            if num_tp is None or num_base is None or num_tp != num_base:
                if str_diff:
                    str_diff += f"\n{key}: {value} != {base_key}: {expected_value}"
                else:
                    str_diff = f"{key}: {value} != {base_key}: {expected_value}"
                mismatches.append((key, value, expected_value))
            
    return len(mismatches) == 0, str_diff


def process_file(input_path, matched_path, mismatched_path, all_path, train_data_clear_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    matched = []
    mismatched = []
    all_data = []
    train_data_clear = []

    for entry in data:
        if not entry.get('ВТовареПоставкиСодержитсяКоличестваВПотребитУпаковке', False):
            all_data.append(entry)
            continue   
        entry_copy = entry.copy() 
        comp, str_diff = compare_numeric_fields(entry)  
        if comp:
            matched.append(entry)
        else:
            entry_copy['СтрокаРазличий'] = str_diff
            mismatched.append(entry_copy)
        all_data.append(entry)
        clear_entry = entry.copy()
        del clear_entry['ИсходныеДанные']
        train_data_clear.append(clear_entry)
      
    with open(matched_path, 'w', encoding='utf-8') as f_out:
        json.dump(matched, f_out, ensure_ascii=False, indent=2)

    with open(mismatched_path, 'w', encoding='utf-8') as f_out:
        json.dump(mismatched, f_out, ensure_ascii=False, indent=2)

    with open(all_path, 'w', encoding='utf-8') as f_out:
        json.dump(all_data, f_out, ensure_ascii=False, indent=2)
    
    with open(train_data_clear_path, 'w', encoding='utf-8') as f_out:
        json.dump(train_data_clear, f_out, ensure_ascii=False, indent=2)

    print(f'Обработано записей: {len(all_data)}')
    print(f'Совпавших записей: {len(matched)}')
    print(f'Не совпавших записей: {len(mismatched)}')


# Пример использования
if __name__ == '__main__':
    process_file(
        input_path='./data/all_1.json',
        matched_path='./data/pre/matched.json',
        mismatched_path='./data/pre/mismatched.json',
        all_path='./data/all_2.json',
        train_data_clear_path='./data/train_data_clear.json'
    )
