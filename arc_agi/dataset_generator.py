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

# Setup OpenAI API key
api_key = "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"

client = OpenAI(api_key=api_key)

def convert_numpy_to_string(arr):
    """Convert a NumPy array to a clean string representation."""
    arr_str = np.array2string(arr, separator=", ", threshold=np.inf, max_line_width=np.inf)
    return f"np.array({arr_str})"
# Function to generate a function from the input and output
def generate_prompt(input_grids, output_grids):
    base_prompt = f"""
    You are an expert ARC (Abstraction and Reasoning Corpus) problem solver. Your task is to transform the input grid into the correct output grid based on the patterns observed. You will be provided with an input grid in the form of a NumPy array, and you should output a transformed NumPy array that matches the expected output based on the learned transformations.
    You will be given a series of input and output grids. Formatted as a python function.
    Note only return this ->
    START FUNC
    def transform(input_grid):
        # Example transformation logic
        output_grid = input_grid + 6
        return output_grid
    END FUNC
    Please try to creatre a function that generalizes to different inputs.
    """
    grid_prompt = []
    for i in range(len(input_grids)):
        grid_prompt.append(f"""Input :""")
        grid_prompt.append(convert_numpy_to_string(input_grids[i]))
        grid_prompt.append(f"""Output :""")
        grid_prompt.append(convert_numpy_to_string(output_grids[i]))
    grid_prompt = "\n".join(grid_prompt)
    prompt = base_prompt + grid_prompt
    return prompt
    
def generate_functions(i):
    id_func_dict = {}
    cn = 0
    for key in train.keys():
        input_grids = []
        output_grids = []
        for elem in train[key]['train']:
            input_grid = elem['input']
            output_grid = elem['output']
            # Convert to np.array
            input_grid = np.array(input_grid)
            output_grid = np.array(output_grid)
            input_grids.append(input_grid)
            output_grids.append(output_grid)
        # Generate the function
        prompt = generate_prompt(input_grids, output_grids)
        try:
            # Call OpenAI API
            response = client.responses.create(
                model="gpt-4.1",
                input=prompt,
            )
            function_code = response.output[0].content[0].text
            # Extract the function
            function_code = function_code.split("START FUNC")[1].split("END FUNC")[0]
            # Save the function with the corresponding id
            id_func_dict[key] = function_code
            print(f"Generated function for ID {key}:")
            print(function_code)
            print("Count:", cn)
            print(
                "Thread ID:", i,
            )
        except Exception as e:
            print(f"Error generating function for ID {key}: {e}")
            # Sleep for a while to avoid hitting the rate limit
            time.sleep(5)
        cn += 1
    # Save the dictionary to a JSON file
    with open(f'arc_agi_functions_{i}.json', 'w') as f:
        json.dump(id_func_dict, f)
    # Sleep for a while to avoid hitting the rate limit
    time.sleep(5)



# Multithreaded function to generate functions
def threaded_generate_functions(start, end):
    threads = []
    for i in range(start, end):
        thread = threading.Thread(target=generate_functions, args=(i,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    print("All threads completed.")

# Call the function to generate functions
threaded_generate_functions(0, 10)