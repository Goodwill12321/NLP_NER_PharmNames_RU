from datasets import load_dataset, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import torch

# Настройки
MODEL_NAME = "google/flan-t5-small"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256

# Загрузка данных
dataset = load_dataset("json", data_files={"train": "data/train_flan_t5.json"}, split="train")

# Токенизатор и модель
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Токенизация
def preprocess(example):
    inputs = tokenizer(example["input"], max_length=MAX_INPUT_LENGTH, padding="max_length", truncation=True)
    labels = tokenizer(example["output"], max_length=MAX_TARGET_LENGTH, padding="max_length", truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# Аргументы обучения
args = Seq2SeqTrainingArguments(
    output_dir="./flan_t5_medsplit",
    evaluation_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
)

# Тренировка
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model),
)

trainer.train()
