'''
This script will generate a dataset of prompts and outputs for the ARC (Abstraction and Reasoning Challenge) using a pre-trained model.

Essentially we will query open ai using arc agi data and will expect an output function. We will gather close to 10,000 outputs. 

The prompt will be as follows(with extra info at the beggining)
input : np.array([
    [3, 2],
    [7, 8]
]) output : np.array([
    [9, 8],
    [13, 14]
])... 
 The output from the model should be a function that takes the input and returns the output.
 START FUNC 
 def transform(input_grid):
     # Example transformation logic
     output_grid = input_grid + 6
    return output_grid
END FUNC

Then we will take the output function parse it and map the id of the arc agi task to the function.
We will save 1000 of these functions in a json file. An iteration will be 1000 tasks.
'''
import os
import json
import random
import time
import requests
import numpy as np
from tqdm import tqdm
import openai
from openai import OpenAI
import threading

with open('arc-agi_training_challenges.json') as f:
    train = json.load(f)
with open('arc-agi_training_solutions.json') as f:
    test = json.load(f)
# Setup OpenAI API key
api_key = ""

client = OpenAI(api_key=api_key)

# Converts np.array to clean string for prompt
def convert_numpy_to_string(arr):
    return np.array2string(arr, separator=', ', threshold=10000)

# === Step 1: Instruction generation prompt ===
def generate_instruction_prompt(input_grids, output_grids, test_input_grid):
    base_prompt = """
You are an expert ARC (Abstraction and Reasoning Corpus) task solver.
Your job is to write **step-by-step transformation instructions** that turn an input grid into its output grid. Make the instructions as clear and detailed as possible, and abstract enough to apply to generalized grids.
Label them 1, 2, 3, etc. and use the following format:
Only return:
START2
1. ...
2. ...
END2

Below are example input/output grids.
"""
    examples = []
    for inp, out in zip(input_grids, output_grids):
        examples.append("INPUT GRID")
        examples.append(convert_numpy_to_string(inp))
        examples.append("OUTPUT GRID")
        examples.append(convert_numpy_to_string(out))
    examples.append("Input test:")
    examples.append(convert_numpy_to_string(test_input_grid))
    examples.append("Output test:")
    return base_prompt + "\n".join(examples)

# === Step 2: Follow instruction with test input ===
def apply_instruction_prompt(test_input, instruction):
    base_prompt = """
You are a Python agent that follows grid transformation instructions.
Take the input grid and apply the steps precisely to generate the output grid.
Only return:
START2
<resulting grid>
END2
"""
    return base_prompt + f"""
INPUT GRID
{convert_numpy_to_string(test_input)}

INSTRUCTION
{instruction}

Output:
"""

# === Step 3: GPT call mockup (replace this with your actual GPT client) ===
def call_gpt(prompt):
    # Replace this with your real GPT API call
    # For OpenAI: openai.ChatCompletion.create(...)
    response = client.responses.create(model="gpt-4.1", input=prompt)
    return response.output[0].content[0].text

# === Step 4: Per-thread function ===
def generate_instruction_and_output(i, chunk_size):
    result_dict = {}
    keys = list(train.keys())
    keys = keys[i:i + chunk_size]
    for key in tqdm(keys, desc=f"Processing {i}"):
        try:
            input_grids = [np.array(x["input"]) for x in train[key]["train"]]
            output_grids = [np.array(x["output"]) for x in train[key]["train"]]
            test_input_grid = np.array(train[key]["test"][0]["input"])
            test_output_grid = np.array(test[key][0]) if key in test else None

            # 1. Generate instruction
            instruction_prompt = generate_instruction_prompt(input_grids, output_grids, test_input_grid)
            instruction_raw = call_gpt(instruction_prompt)

            if "START2" not in instruction_raw or "END2" not in instruction_raw:
                print(f"[{key}] No valid instruction.")
                continue

            instruction = instruction_raw.split("START2")[1].split("END2")[0].strip()

            # 2. Use instruction on test input
            apply_prompt = apply_instruction_prompt(test_input_grid, instruction)
            output_raw = call_gpt(apply_prompt)

            if "START2" not in output_raw or "END2" not in output_raw:
                print(f"[{key}] No valid output grid.")
                continue

            predicted_grid = output_raw.split("START2")[1].split("END2")[0].strip()
            result_dict[key] = {
                "instruction": instruction,
                "test_input": test_input_grid.tolist(),
                "predicted_output": predicted_grid,
                "expected_output": test_output_grid.tolist() if test_output_grid is not None else None
            }

        except Exception as e:
            print(f"[{key}] Error: {e}")
            time.sleep(2)

    with open(f"arc_agi_instructions_{i}.json", "w") as f:
        json.dump(result_dict, f)
    time.sleep(2)

# === Step 5: Multithreaded execution ===
def threaded_instruction_generation(start, end):
    threads = []
    keys = list(train.keys())
    # Divide into end - start chunks
    chunk_size =  len(keys) // (end - start)
    start = 0
    for i in range(start, end):
        thread = threading.Thread(target=generate_instruction_and_output, args=(start, chunk_size))
        threads.append(thread)
        thread.start()
        start += chunk_size
    for thread in threads:
        thread.join()
    print("All threads done.")

# === Run ===
threaded_instruction_generation(0, 10)