import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# === Настройки ===
model_name = "ai-forever/FRED-T5-base"   # можно попробовать "ai-forever/FRED-T5-large"
input_file = "./training/data/train_data_clear.data"

# === Загрузка модели и токенайзера ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# === Токенизация ===
def preprocess(example):
    model_inputs = tokenizer(
        example["input"],
        max_length=256,
        padding="max_length",
        truncation=True
    )
    labels = tokenizer(
        example["output"],
        max_length=512,
        padding="max_length",
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# === Метрики ===
rouge = load_metric("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Убираем лишние пробелы
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    # Берем RougeL как основную метрику
    return {"rougeL": result["rougeL"].mid.fmeasure}


def main():
    # === Загрузка данных ===
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    records = []
    for item in raw_data:
        product = item.get("ТоварПоставки", "")
        input_parts = [
            "Задание: Извлеки части из названия лекарственного препарата.",
            f"Наименование препарата: {product}",
        ]
        input_text = "\n".join(input_parts)

        output_dict = {
            k: str(v) for k, v in item.items()
            if v not in [None, ""] and k not in ["ТоварПоставки", "ПредставлениеТовара", "ГУИД_Записи"]
        }
        output_text = json.dumps(output_dict, ensure_ascii=False)

        records.append({
            "input": input_text,
            "output": output_text
        })

    # === Разделение ===
    df = pd.DataFrame(records)
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    print("Размер train:", len(train_dataset))
    print("Размер test:", len(test_dataset))

    train_dataset = train_dataset.map(preprocess, remove_columns=["input", "output"])
    test_dataset = test_dataset.map(preprocess, remove_columns=["input", "output"])

    # === Параметры обучения ===
    training_args = TrainingArguments(
        output_dir="./fred-t5-med-ner",
        per_device_train_batch_size=2,        # безопасно для 12Гб VRAM
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,        # эквивалент batch=16
        num_train_epochs=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=20,
        load_best_model_at_end=True,
        save_total_limit=3,
        fp16=True,   # mixed precision
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics
    )

    # === Запуск ===
    trainer.train()

    # === Сохранение модели ===
    model.save_pretrained("./fred-t5-med-ner")
    tokenizer.save_pretrained("./fred-t5-med-ner")
    print("Final training loss:", trainer.state.log_history[-1])

if __name__ == '__main__':
    main()
