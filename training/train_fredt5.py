import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
import evaluate


# === Настройки модели ===
MODEL_NAME = "ai-forever/FRED-T5-large"
INPUT_FILE = "./training/data/train_data_clear.data"


# === Токенизация ===
def preprocess(example, tokenizer, max_input_len=256, max_output_len=512):
    """Подготовка токенов для модели"""
    model_inputs = tokenizer(
        example["input"], max_length=max_input_len, padding="max_length", truncation=True
    )
    labels = tokenizer(
        example["output"], max_length=max_output_len, padding="max_length", truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# === Метрика ===
rouge = evaluate.load("rouge")


def compute_metrics(eval_pred, tokenizer):
    """Подсчёт качества по ROUGE-L"""
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    result = rouge.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    return {"rougeL": result["rougeL"].mid.fmeasure}


# === Загрузка и подготовка данных ===
def load_dataset(tokenizer):
    """Загрузка исходных данных и преобразование в HuggingFace Dataset"""
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
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
            k: str(v)
            for k, v in item.items()
            if v not in [None, ""]
            and k not in ["ТоварПоставки", "ПредставлениеТовара", "ГУИД_Записи"]
        }
        output_text = json.dumps(output_dict, ensure_ascii=False)

        records.append({"input": input_text, "output": output_text})

    # Разделение train/test
    df = pd.DataFrame(records)
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    train_dataset = train_dataset.map(
        lambda x: preprocess(x, tokenizer), remove_columns=["input", "output"]
    )
    test_dataset = test_dataset.map(
        lambda x: preprocess(x, tokenizer), remove_columns=["input", "output"]
    )

    return train_dataset, test_dataset


# === Основной процесс ===
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    train_dataset, test_dataset = load_dataset(tokenizer)

    print("Размер train:", len(train_dataset))
    print("Размер test:", len(test_dataset))
    print("Пример:", test_dataset[0])

    training_args = TrainingArguments(
        output_dir="./fredt5-large-ner",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  # эффективный batch = 8
        num_train_epochs=5,
        eval_strategy="steps",   
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=20,
        load_best_model_at_end=True,
        save_total_limit=2,
        fp16=True,  # экономия памяти
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    trainer.train()

    # Сохранение
    model.save_pretrained("./fredt5-large-ner")
    tokenizer.save_pretrained("./fredt5-large-ner")
    print("✅ Обучение завершено и модель сохранена!")


if __name__ == "__main__":
    main()
