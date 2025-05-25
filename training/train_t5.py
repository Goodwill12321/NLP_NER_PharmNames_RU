import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq

# Путь к файлу с данными
input_file = "./data/train.jsonl"

# Загрузка JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Список целевых полей
target_fields = [
    "ТорговоеНаименование_ТП",
    "Дозировка_ТП",
    "Лекформа_ТП",
    "ПервичнаяУпаковкаНазвание_ТП",
    "ПервичнаяУпаковкаКоличество_ТП",
    "ПотребительскаяУпаковкаКолво_ТП",
    "ВторичнаяУпаковкаНазвание_ТП",
    "ВторичнаяУпаковкаКоличество_ТП"
]

# Формируем датафрейм
records = []
for arr in data:
    for item in arr:
        hints = item.get("ИсходныеДанные", {})
        input_parts = [
            f"Задание: Извлеки части наименования из товара.",
            f"Product: {item['ТоварПоставки']}",
            f"Hints: "
        ]
        input_parts.extend([
            f"{k}: {v}" for k, v in hints.items()
        ])
        input_text = "\n".join(input_parts)

        # Упорядоченный JSON-ответ
        output_dict = {}
        for field in target_fields:
            val = item.get(field)
            output_dict[field] = val if val else "null"

        output_text = json.dumps(output_dict, ensure_ascii=False)

        records.append({
            "input": input_text,
            "output": output_text
        })

df = pd.DataFrame(records)

# Разделение train/test
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Инициализация токенизатора и модели
model_name = "cointegrated/rut5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Токенизация
max_input_len = 512
max_output_len = 128

def preprocess(example):
    inputs = tokenizer(example["input"], max_length=max_input_len, padding="max_length", truncation=True)
    targets = tokenizer(example["output"], max_length=max_output_len, padding="max_length", truncation=True)
    inputs["labels"] = targets["input_ids"]
    return inputs

train_dataset = train_dataset.map(preprocess, remove_columns=["input", "output"])
test_dataset = test_dataset.map(preprocess, remove_columns=["input", "output"])

# Аргументы обучения
training_args = TrainingArguments(
    output_dir="./t5-med-ner",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=20,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=False  # ставь True, если есть GPU с поддержкой
)

# Коллатор
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Тренер
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Обучение
trainer.train()

# Сохранение
model.save_pretrained("./t5-med-ner")
tokenizer.save_pretrained("./t5-med-ner")
