import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType

def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == '__main__':
    # --- Configuration for Multilingual Model ---
    MODEL_NAME = "xlm-roberta-base" # Use the multilingual model
    OUTPUT_DIR = "models/abuse-detector-xlm-roberta-lora" # New output folder
    
   
    DATA_FILES = {
        "train": "data/processed/abuse_train.csv",
        "validation": "data/processed/abuse_val.csv"
    }
    NUM_EPOCHS = 1

    try:
        raw_datasets = load_dataset('csv', data_files=DATA_FILES)
    except FileNotFoundError:
        print("Error: Processed data not found. Please run src/data_preprocessing.py first.")
        exit()

    print("Using a smaller subset for training.")
    raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42).select(range(10000))
    raw_datasets["validation"] = raw_datasets["validation"].shuffle(seed=42).select(range(1000))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_datasets = raw_datasets.map(lambda examples: tokenize_function(tokenizer, examples), batched=True)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["query", "value"],
        bias="none",
    )

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting multilingual LoRA model training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Training complete. Multilingual adapters saved to {OUTPUT_DIR}")