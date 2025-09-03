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
import numpy as np

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

import numpy as np

def print_token_stats(dataset, tokenizer):
    inputs = [len(tokenizer(x)["input_ids"]) for x in dataset["input"]]
    outputs = [len(tokenizer(x)["input_ids"]) for x in dataset["output"]]

    print("Макс длина входа:", max(inputs))
    print("95 перцентиль входа:", np.percentile(inputs, 95))
    print("Макс длина выхода:", max(outputs))
    print("95 перцентиль выхода:", np.percentile(outputs, 95))
    print("Пример входа:", dataset["input"][0])
    print("Пример выхода:", dataset["output"][0])



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

    #print_token_stats(train_dataset, tokenizer)
    #print_token_stats(test_dataset, tokenizer)
        # >>> вот здесь считаем статистику <<<
    print("📊 Train:")
    print_token_stats(train_dataset, tokenizer)
    print("📊 Test:")
    print_token_stats(test_dataset, tokenizer)

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
    #print("Пример:", test_dataset[0])


    inputs = [len(tokenizer(x)["input_ids"]) for x in train_dataset["input"]]
    outputs = [len(tokenizer(x)["input_ids"]) for x in test_dataset["output"]]
    print("Max input:", max(inputs), "Max output:", max(outputs), "95th percentile input:", np.percentile(inputs, 95), "95th percentile output:", np.percentile(outputs, 95))

 
    MAX_INPUT = 128
    MAX_OUTPUT = 256
    BATCH_SIZE = 1
    EVAL_STEPS = 500
    SAVE_STEPS = 500
    LOGGING_STEPS = 100
    LEARNING_RATE = 5e-5
    EPOCHS = 5

    #import transformers
    #print("Transformers version:", transformers.__version__)

    #from transformers import TrainingArguments
    #print("TrainingArguments из:", TrainingArguments.__module__)


    training_args = TrainingArguments(
        output_dir="./fredt5-large-ner",
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=True
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
