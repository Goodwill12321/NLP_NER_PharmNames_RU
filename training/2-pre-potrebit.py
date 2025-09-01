import re
import json

# --- Настройки ---
ALLOWED_PREFIXES = ['№', 'N', 'x', 'X', 'х', 'Х']
INPUT_FILE = './data/train_nlp.json'
OUTPUT_WITH = './data/pre/с_количеством.json'
OUTPUT_WITHOUT = './data/pre/без_количества.json'
OUTPUT_ALL = './data/all_1.json'
ERROR_FILE = './data/pre/errors_potreb.txt'

def extract_pack_form(text, amount):
    """Ищем паттерн типа '№ 10', 'N10', 'x10' в строке."""
    if not amount:
        return None

    for prefix in ALLOWED_PREFIXES:
        # Пробел может быть, а может и нет
        pattern = re.escape(prefix) + r'\s*' + re.escape(str(amount)) + r'\b'
        match = re.search(pattern, text)
        if match:
            return match.group()
    return None


def process_dataset(filepath):
    try:
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        with open(ERROR_FILE, 'w', encoding='utf-8') as ef:
            ef.write(f"Ошибка JSON при чтении файла: {e}\n")
        raise RuntimeError(f"Файл {filepath} не является корректным JSON.")

    with_qty = []
    without_qty = []

    for i, entry in enumerate(data, 1):
        #qty = entry.get('ПотребительскаяУпаковкаКолво_ТП', '').strip()
        raw_qty = entry.get('ПотребительскаяУпаковкаКоличество_ТП', '')
        qty = str(raw_qty).strip() if raw_qty is not None else ''
        has_qty = bool(qty)
        # Куда положить
        if has_qty:
            # Попробуем найти полное выражение количества
            pack_form = extract_pack_form(entry.get('ТоварПоставки', ''), qty)
            if pack_form:
                entry['ПотребительскаяУпаковка_ТП'] = pack_form
            entry['ВТовареПоставкиСодержитсяКоличестваВПотребитУпаковке'] = True
            with_qty.append(entry)
        else:
            # Попробуем всё равно найти потенциальное количество, если есть
            text_good = entry.get('ТоварПоставки', '')
            if not text_good:
                continue
            pattern = r'(' + '|'.join(map(re.escape, ALLOWED_PREFIXES)) + r')\s*(\d{1,4})(?![a-zA-Zа-яА-Я])'

            possible_number_match = re.search(pattern, text_good)
            if possible_number_match:
                number = possible_number_match.group(2)
                full_form = extract_pack_form(text_good, number)
                if full_form:
                    entry['ПотребительскаяУпаковка_ТП'] = full_form
                entry['ВТовареПоставкиСодержитсяКоличестваВПотребитУпаковке'] = True 
                with_qty.append(entry)      
            else:
                entry['ВТовареПоставкиСодержитсяКоличестваВПотребитУпаковке'] = False 
                without_qty.append(entry)

    return with_qty, without_qty


def save_jsonl(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def save_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def append_json(data, filepath):
    with open(filepath, 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)        


if __name__ == '__main__':
    with_qty, without_qty = process_dataset(INPUT_FILE)
    save_json(with_qty, OUTPUT_WITH)
    save_json(without_qty, OUTPUT_WITHOUT)
    all = with_qty + without_qty
    save_json(all, OUTPUT_ALL)
    #append_json(without_qty, OUTPUT_ALL)
    
    print(f"✅ Записей с количеством: {len(with_qty)}")
    print(f"❌ Записей без количества: {len(without_qty)}")
