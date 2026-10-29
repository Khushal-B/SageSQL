"""Fine-tune a LoRA adapter on Qwen3-4B using (schema, question, SQL) triples."""
import json

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

from sqlsage.generator import BASE_MODEL, PROMPT

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
))

# data/train.json: [{"schema": "...", "question": "...", "sql": "..."}, ...]
rows = json.load(open("data/train.json"))


def encode(row):
    text = PROMPT.format(schema=row["schema"], question=row["question"]) + row["sql"]
    return tokenizer(text, truncation=True, max_length=1024)


ds = Dataset.from_list(rows).map(encode, remove_columns=["schema", "question", "sql"])

trainer = Trainer(
    model=model,
    args=TrainingArguments("outputs/lora_adapter", num_train_epochs=3,
                           per_device_train_batch_size=4, learning_rate=2e-4, logging_steps=20),
    train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()
model.save_pretrained("outputs/lora_adapter")
