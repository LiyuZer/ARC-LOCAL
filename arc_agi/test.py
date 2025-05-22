# Firstly we will import the necessary libraries 
import openai
import random   
import time
import os
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch
from huggingface_hub import login
from black import format_str, FileMode
from peft import get_peft_model, LoraConfig, TaskType
import traceback
from trl import PPOTrainer, PPOConfig
from trl import AutoModelForCausalLMWithValueHead

# === Data loading ===
with open('arc-agi_training_challenges.json') as f:
    train = json.load(f)
with open('arc-agi_training_solutions.json') as f:
    test = json.load(f)

model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"

# === HF login ===
login(token="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")

# === Model / tokenizer ===

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=True,
)

# LoRA
lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.05)
peft_model = get_peft_model(base_model, lora_config)

# Wrap with value head
model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model)

# === PPO trainer ===
ppo_config = PPOConfig(learning_rate=1e-5, batch_size=1, mini_batch_size=1)
ppo_trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer)

# === Prepare ARC sample ===
inputs = [np.array(item["input"]) for item in train["00576224"]["train"]]
outputs = [np.array(item["output"]) for item in train["00576224"]["train"]]

test_inputs = [np.array(item["input"]) for item in train["00576224"]["test"]]
test_outputs = [np.array(item) for item in test["00576224"]]

# === Helpers ===

def fmt(arr):
    return f"{arr.tolist()}"


def base_prompt(inputs_, outputs_):
    prompt_lines = [
        "Identify the pattern in the input and output grids in bullet points. Go for more complex patterns.",
        """
INPUT GRID PATTERN 1
    [[1, 2],
    [3, 4]]
OUTPUT GRID PATTERN 1
    [[5, 6, 5,6],
    [7, 8, 7,8]]
Output test:
You must output the the instructions in this format(Start 2 and End 2 for the next pattern):
START1
1. Add 2 to each element in the input grid.
2. Then, extend the grid by repeating the first and the second rows, along the same dimension.
END1

This is just a random example, the next pattern.
""",
    ]
    for inp, out in zip(inputs_, outputs_):
        prompt_lines.append("INPUT GRID PATTERN 2")
        prompt_lines.append(fmt(inp))
        prompt_lines.append("OUTPUT GRID PATTERN 2")
        prompt_lines.append(fmt(out))
    prompt_lines.append("Input test:")
    prompt_lines.append(fmt(test_inputs[0]))
    prompt_lines.append("Output test:")
    return "\n".join(prompt_lines)


# === Generation utilities ===

def create_instructions(inputs_, outputs_):
    prompt = base_prompt(inputs_, outputs_)
    device = next(model.parameters()).device
    inputs_tok = tokenizer(prompt, return_tensors="pt").to(device)
    outputs_tok = model.generate(**inputs_tok, max_length=1000, do_sample=True, temperature=0.7)
    generated_text = tokenizer.decode(outputs_tok[0], skip_special_tokens=True)

    if "START2" not in generated_text or "END2" not in generated_text:
        return None, generated_text

    start = generated_text.find("START2") + len("START2")
    end = generated_text.find("END2")
    instruction = generated_text[start:end].strip()
    return instruction, generated_text


def instruction_prompt(test_input, instruction):
    prompt_lines = [
        "Apply the instruction to the input grid and generate the output grid.",
        """
INPUT GRID PATTERN 1
    [[1, 2],
    [3, 4]]

INSTRUCTION
1. Add 2 to each element in the input grid.
2. Then, extend the grid by repeating the first and the second rows, along the same dimension.

You must output the the arrays in this format(Start 2 and End 2 for the next pattern):
START1
[[5, 6, 5, 6],
[7, 8, 7, 8]]
END1

This is just a random example, the next pattern.
""",
    ]
    prompt_lines.append("INPUT GRID PATTERN 2")
    prompt_lines.append(fmt(test_input))
    prompt_lines.append("INSTRUCTION")
    prompt_lines.append(instruction)
    return "\n".join(prompt_lines)


def follow_instructions(test_input, instruction):
    prompt = instruction_prompt(test_input, instruction)
    device = next(model.parameters()).device
    inputs_tok = tokenizer(prompt, return_tensors="pt").to(device)
    outputs_tok = model.generate(**inputs_tok, max_length=1000, do_sample=True, temperature=0.7)
    generated_text = tokenizer.decode(outputs_tok[0], skip_special_tokens=True)

    if "START2" not in generated_text or "END2" not in generated_text:
        return None, generated_text

    core = generated_text.split("END2")[0].split("START2")[1].strip()
    return core, generated_text


# === RL helpers ===

def reward_function(pred, expected):
    correct = np.sum(pred == expected)
    return correct / pred.size



def step_ppo_trainer(prompt, response, reward):
    device = next(model.parameters()).device
    query_tensor = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(device)
    response_tensor = tokenizer(response, return_tensors="pt", truncation=True).input_ids.to(device)
    reward_tensor = [torch.tensor([reward], dtype=torch.float32).to(device)]

    ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward_tensor)

# === Training loop ===

num_steps = 100

def train_loop():
    for _ in range(num_steps):
        instruction, instr_text = create_instructions(inputs, outputs)
        base_prompt_str = base_prompt(inputs, outputs)

        if instruction is None:
            step_ppo_trainer(base_prompt_str, instr_text, 0.0)
            print("No instruction generated – stepped with zero reward.")
            continue

        grid_text, grid_response_text = follow_instructions(test_inputs[0], instruction)
        instr_prompt_str = instruction_prompt(test_inputs[0], instruction)

        if grid_text is None:
            step_ppo_trainer(instr_prompt_str, grid_response_text, 0.0)
            print("No output grid generated – stepped with zero reward.")
            continue

        pred_grid = np.array(eval(grid_text))
        expected_grid = np.array(test_outputs[0])
        reward = reward_function(pred_grid, expected_grid)
        print("Reward:", reward)

        step_ppo_trainer(base_prompt_str, instr_text, reward)


if __name__ == "__main__":
    train_loop()
