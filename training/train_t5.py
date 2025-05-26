import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq

# Нормализация ключей: заменим все пробелы и приведём к нижнему регистру
def normalize_key(key):
    key = key.replace(" ", "_").replace("-", "_").lower()
    key = re.sub(r"__+", "_", key)
    return key.strip("_")

# Целевые поля (нормализованные имена)
target_fields = [
    "торговое_наименование_тп",
    "дозировка_тп",
    "лек_форма_тп",
    "первичная_упаковка_название_тп",
    "первичная_упаковка_количество_тп",
    "потребительская_упаковка_количество_тп",
    "вторичная_упаковка_название_тп",
    "вторичная_упаковка_количество_тп"
]

# Путь к файлу с исходными данными
input_file = "./data/train.jsonl"

# Загружаем JSON
with open(input_file, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

records = []

for array in raw_data:
    for item in array:
        # Нормализуем ключи (приводим к одному стилю)
        item_norm = {normalize_key(k): v for k, v in item.items()}

        # Основные поля
        product = item_norm.get("товар_поставки", item_norm.get("товарпоставки", ""))
        
        # Подсказки из эталонного представления
        hints = {}
        for k in ["торговое_наименование", "дозировка", "лек_форма", "первичная_упаковка_название", "первичная_упаковка_количество", "потребительская_упаковка_количество", "вторичная_упаковка_название", "вторичная_упаковка_количество"]:
            v = item_norm.get(k)
            if v:
                hints[k] = str(v)

        # Вход
        input_parts = [
            "Задание: Извлеки части наименования из товара.",
            f"Product: {product}",
            "Hints:"
        ]
        input_parts += [f"{k}: {v}" for k, v in hints.items()]
        input_text = "\n".join(input_parts)

        # Выход
        output_dict = {}
        for field in target_fields:
            value = item_norm.get(field)
            output_dict[field] = str(value) if value is not None else "null"
        
        output_text = json.dumps(output_dict, ensure_ascii=False)

        records.append({
            "input": input_text,
            "output": output_text
        })

# В датафрейм и train/test
df = pd.DataFrame(records)
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# T5 модель
model_name = "cointegrated/rut5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Токенизация
def preprocess(example):
    model_inputs = tokenizer(example["input"], max_length=512, padding="max_length", truncation=True)
    labels = tokenizer(example["output"], max_length=128, padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(preprocess, remove_columns=["input", "output"])
test_dataset = test_dataset.map(preprocess, remove_columns=["input", "output"])

# Обучение
training_args = TrainingArguments(
    output_dir="./t5-med-ner",
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

# Запуск обучения
trainer.train()

# Сохранение модели
model.save_pretrained("./t5-med-ner")
tokenizer.save_pretrained("./t5-med-ner")
