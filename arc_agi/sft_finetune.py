import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login

# === Login to HF Hub (optional) ===
login(token="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")

# === Config ===
model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"
dataset_path = "arc_instruction_proposal_dataset"  # or "arc_instruction_interpreter_dataset"
use_lora = True
output_dir = "finetuned_qwen_proposer"  # Change for interpreter

# === Load Dataset ===
dataset = load_from_disk(dataset_path)

# === Load Tokenizer and Base Model ===
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# === Apply LoRA (optional) ===
if use_lora:
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05
    )
    model = get_peft_model(model, lora_config)

# === Preprocessing Function ===
def tokenize(example):
    # Add a separator between prompt and completion if needed
    input_text = example["prompt"]
    target_text = example["completion"]
    full_text = input_text + "\n" + target_text

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# === Data Collator ===
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    evaluation_strategy="no",
    report_to="none"
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=collator
)

# === Train ===
trainer.train()

# === Save Model ===
trainer.save_model(output_dir)
print(f"✅ Model saved to {output_dir}")
