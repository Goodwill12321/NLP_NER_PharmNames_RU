import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq

# === Утилиты ===


# T5 модель
model_name = "cointegrated/rut5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Токенизация
def preprocess(example):
    model_inputs = tokenizer(example["input"], max_length=128, padding="max_length", truncation=True)
    labels = tokenizer(example["output"], max_length=256, padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    # === Загрузка данных ===

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
            "Задание: Извлеки части из названия лекарственного препарата.",
            f"Наименование препарата: {product}",
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

    max_input_len = max([len(tokenizer(r["input"])["input_ids"]) for r in records])
    max_output_len = max([len(tokenizer(r["output"])["input_ids"]) for r in records])
    print("Максимальная длина input:", max_input_len)
    print("Максимальная длина output:", max_output_len)

    df = pd.DataFrame(records)
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    train_dataset = train_dataset.map(preprocess, remove_columns=["input", "output"])
    test_dataset = test_dataset.map(preprocess, remove_columns=["input", "output"])
    print(test_dataset[0])
    # === Обучение ===

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

    # === Запуск ===
    trainer.train()

    # === Сохранение модели ===
    model.save_pretrained("./t5-med-ner")
    tokenizer.save_pretrained("./t5-med-ner")

if __name__ == '__main__':
    main()