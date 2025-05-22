import os
import json
import random
import time
from datasets import Dataset

# === Load instruction files ===
base = 'arc_agi_instructions'
paths = [f'{base}_{i}.json' for i in range(0, 901, 100)]

json_files = {}
for path in paths:
    with open(path, 'r') as f:
        json_files.update(json.load(f))

# === Load ARC training inputs ===
with open('arc-agi_training_challenges.json') as f:
    train = json.load(f)

# === Augment json_files with training input/output ===
for key in json_files.keys():
    if key in train:
        if 'training' not in json_files[key]:
            json_files[key]['training'] = {}
        json_files[key]['training']['input'] = [item["input"] for item in train[key]['train']]
        json_files[key]['training']['output'] = [item["output"] for item in train[key]['train']]

print("✅ Augmented ARC task sample:")
print(json.dumps(json_files[list(json_files.keys())[0]], indent=2))

# === Build proposal dataset (Propose instructions from (input, output) pairs) ===
proposal_data = []

for key, item in json_files.items():
    train_inputs = item.get("training", {}).get("input", [])
    train_outputs = item.get("training", {}).get("output", [])
    instruction = item.get("instruction", "")

    if len(train_inputs) == 0 or len(train_outputs) == 0 or not instruction:
        continue

    prompt_lines = []
    for inp, out in zip(train_inputs, train_outputs):
        prompt_lines.append("INPUT GRID")
        prompt_lines.append(str(inp))
        prompt_lines.append("OUTPUT GRID")
        prompt_lines.append(str(out))
    prompt = "\n".join(prompt_lines)

    completion = f"START1\n{instruction}\nEND1"
    proposal_data.append({
        "id": key,
        "prompt": prompt,
        "completion": completion,
    })

proposal_dataset = Dataset.from_list(proposal_data)
proposal_dataset.save_to_disk("arc_instruction_proposal_dataset")
proposal_dataset.to_json("arc_instruction_proposal_dataset.json")
print("✅ Saved proposal dataset.")

# === Build interpreter dataset (Apply instruction to input to get output) ===
interpreter_data = []

for key, item in json_files.items():
    test_input = item.get("test_input", [])
    instruction = item.get("instruction", "")
    output = item.get("expected_output") or item.get("predicted_output")

    if not test_input or not instruction or not output:
        continue

    prompt = (
        f"INPUT GRID\n{test_input}\n"
        f"INSTRUCTION\nSTART1\n{instruction}\nEND1\n\n"
        "Output:"
    )
    completion = f"START1\n{output}\nEND1"

    interpreter_data.append({
        "id": key,
        "prompt": prompt,
        "completion": completion,
    })

interpreter_dataset = Dataset.from_list(interpreter_data)
interpreter_dataset.save_to_disk("arc_instruction_interpreter_dataset")
interpreter_dataset.to_json("arc_instruction_interpreter_dataset.json")
print("✅ Saved interpreter dataset.")
