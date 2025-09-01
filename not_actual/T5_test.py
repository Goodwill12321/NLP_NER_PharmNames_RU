from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "./flan_t5_medsplit/checkpoint-final"  # или checkpoint-1000
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

text = "ТоварПоставки: Азитромицин-Эдвансд капс 500мг №3 блист | ПредставлениеТовара: Азитромицин-Эдвансд капсулы, 500 мг, 3 шт. - блистер (1)  - пачка картонная | ТорговоеНаименование: Азитромицин-Эдвансд | Дозировка: 500 мг | ЛекФорма: КАПСУЛЫ | ПервичнаяУпаковкаНазвание: БЛИСТЕР | ПервичнаяУпаковкаКоличество: 3.0 | ВторичнаяУпаковкаНазвание: КАРТОННАЯ ПАЧКА | ВторичнаяУпаковкаКоличество: 1 | ПотребительскаяУпаковкаКолво: 3.0"

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
outputs = model.generate(**inputs, max_length=256)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("🔎 decoded:", decoded)
