import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === Утилиты ===

def to_camel_case(s: str) -> str:
    # Если строка заканчивается на "_ТП", отделим суффикс
    suffix = ''
    if s.lower().endswith('_тп'):
        s, suffix = s[:-3], '_ТП'
    
    # Если в оставшейся строке нет подчёркиваний — считаем, что это уже CamelCase
    if '_' not in s:
        return s.capitalize() + suffix

    # Преобразуем основную часть в CamelCase
    parts = s.split('_')
    camel = parts[0].capitalize() + ''.join(word.capitalize() for word in parts[1:])
    
    return camel + suffix


def normalize_keys_to_camel_case(d: dict) -> dict:
    return {to_camel_case(k): v for k, v in d.items()}

# Целевые поля (имена уже в CamelCase-стиле)
target_fields = [
    "ТорговоеНаименование_ТП",
    "Дозировка_ТП",
    "ЛекФорма_ТП",
    "ПервичнаяУпаковкаНазвание_ТП",
    "ПервичнаяУпаковкаКоличество_ТП",
    "ПотребительскаяУпаковкаКоличество_ТП",
    "ВторичнаяУпаковкаНазвание_ТП",
    "ВторичнаяУпаковкаКоличество_ТП"
]

# === Загрузка данных ===
# T5 модель
model_name = "melmoth/ru-rope-t5-small-instruct"
#tokenizer = T5Tokenizer.from_pretrained(model_name)
#model = T5ForConditionalGeneration.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# Токенизация
def preprocess(example):
    model_inputs = tokenizer(example["input"], max_length=1024, padding="max_length", truncation=True)
    labels = tokenizer(example["output"], max_length=512, padding="longest", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    input_file = "./data/train_data_clear.json"
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    records = []

    # for array in raw_data:
    for item in raw_data:
        # Поддержка вложенной структуры
        # base_data = normalize_keys_to_camel_case(item.get("ИсходныеДанные", {}))
        #flat_item = normalize_keys_to_camel_case(item)

        # Основное поле (Product)
        product = item.get("ТоварПоставки", "")

        # Подсказки — все поля из ИсходныеДанные
        #hints = {k: str(v) for k, v in base_data.items() if v not in [None, ""]}

        input_parts = [
            "Задание: Извлеки части наименования из названия лекарственного препарата или товара фармацевтического назначения.",
            f"Наименование товара: {product}",
            "Описание полей:",
            "ТорговоеНаименование_ТП — торговая марка",
            "Дозировка_ТП — количество, концентрация активного вещества (например, мг, мл, мг/мл и др.).",
            "ЛекФорма_ТП — форма выпуска: таблетки, капсулы, мазь и др.",
            "ПервичнаяУпаковкаНазвание_ТП — вид упаковки: блистер, флакон и др.",
            "ПервичнаяУпаковкаКоличество_ТП — количество единиц в первичной уп.",
            "ПотребительскаяУпаковкаКоличество_ТП — количество единиц в потребительской уп..",
            "ВторичнаяУпаковкаНазвание_ТП — название вторичной уп. (например, коробка).",
            "ВторичнаяУпаковкаКоличество_ТП — количество первичных уп. в ней.",
            "Если какие-либо части не определяются, указывай null.",
            "Поле `ВТовареПоставкиСодержитсяКоличестваВПотребитУпаковке` может указывать, что в названии товара нет данных о потребительской упаковке  - она может быть null",
        ]

        #input_parts += [f"{k}: {v}" for k, v in hints.items()]
        input_text = "\n".join(input_parts)

        # Выход
        #output_dict = {}
        #for field in target_fields:
        #   value = flat_item.get(to_camel_case(field))
        #    output_dict[field] = str(value) if value is not None else "null"
        output_dict = {k: str(v) for k, v in item.items() if v not in [None, ""] and k not in ["ТоварПоставки","ПредставлениеТовара", "ГУИД_Записи"]}
        output_text = json.dumps(output_dict, ensure_ascii=False)

        records.append({
            "input": input_text,
            "output": output_text
        })

    # === Разделение и токенизация ===

    df = pd.DataFrame(records)
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
  
    train_dataset = train_dataset.map(preprocess, remove_columns=["input", "output"])
    test_dataset = test_dataset.map(preprocess, remove_columns=["input", "output"])

    # === Обучение ===

    training_args = TrainingArguments(
        output_dir="./model",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        save_total_limit=2,
        fp16=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )

    # === Запуск ===
    trainer.train()

    # === Сохранение модели ===
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")
    print("Модель успешно обучена и сохранена в ./model")

if __name__ == '__main__':
    main()