
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
import function_dict
import multiprocessing as mp
import queue

base_prompt = """
You generate transformation functions for the ARC (Abstraction and Reasoning Challenge) dataset.

## Signature Types (use 'signature' field):
1: GRID→GRID, 2: GRID→INT, 3: GRID→BOOL, 4: GRID,GRID→GRID, 5: GRID,INT→GRID, 6: GRID,FUNCTION→GRID,
7: LIST_GRID→LIST_GRID, 8: LIST_GRID→GRID, 9: LIST_GRID→INT, 10: LIST_GRID,INT→LIST_GRID,
11: LIST_GRID,FUNCTION→LIST_GRID, 12: LIST_GRID,GRID→GRID, 13: GRID,FUNCTION,FUNCTION→GRID,
14: LIST_GRID,INT,INT→GRID, 15: FUNCTION,INT→FUNCTION

GRID = 2D numpy array, LIST_GRID = list of 2D numpy arrays, INT = int, BOOL = bool, FUNCTION = callable as per above.

- Main function must be between: start1 ... end1
- Test function named test(only test with the above function as an input) must be between: start2 ... end2. This will be used to test the function
we will call like test(function_name), so the function should be named test and have 1 argument.
- Output format: {"name": function_name, "signature": n}
- Test returns: (True, None) if passed, else (False, "reason")
- No imports; numpy as np is assumed. Note grids are 2D numpy arrays.
- Only code within the required start/end markers is valid.
- Functions must eventually return(no infinite loops).

Function description below:
"""
base_prompt_fixer = """
You fix ARC transformation functions to match their type signature and ensure they pass their tests.

## Signature Types (use 'signature' field):
1: GRID→GRID, 2: GRID→INT, 3: GRID→BOOL, 4: GRID,GRID→GRID, 5: GRID,INT→GRID, 6: GRID,FUNCTION→GRID,
7: LIST_GRID→LIST_GRID, 8: LIST_GRID→GRID, 9: LIST_GRID→INT, 10: LIST_GRID,INT→LIST_GRID,
11: LIST_GRID,FUNCTION→LIST_GRID, 12: LIST_GRID,GRID→GRID, 13: GRID,FUNCTION,FUNCTION→GRID,
14: LIST_GRID,INT,INT→GRID, 15: FUNCTION,INT→FUNCTION

- Main function must be between: start1 ... end1
- Test function named test(only test with the above function as an input) must be between: start2 ... end2. This will be used to test the function
we will call like test(function_name), so the function should be named test and have 1 argument.
- Fix code as needed (comment what you fix), keep function/test names.
- Test returns (True, None) if passed, else (False, "reason").
- No imports; numpy as np is assumed. Note grids are 2D numpy arrays.
- Functions must eventually return(no infinite loops).

"""
def run_with_timeout(fn, timeout=3):
    """Runs fn() in a subprocess; kills it if it takes longer than `timeout` seconds."""
    q = mp.Queue()
    def worker():
        try:
            q.put(fn())
        except Exception as e:
            q.put(e)
    p = mp.Process(target=worker)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        raise TimeoutError("Function call timed out")
    result = q.get()
    if isinstance(result, Exception):
        raise result
    return result

# Setup OpenAI API key
api_key = ""
client = OpenAI(
    api_key   = api_key,
    timeout   = 30,      # hard deadline for *any* HTTP round‑trip
    max_retries = 0      # we’ll do our own retry/back‑off
)

def safe_openai_call(prompt, fixer=False):
    model = "gpt-4o-mini"
    body  = base_prompt_fixer + prompt if fixer else base_prompt + prompt
    for tries in range(5):
        try:
            return client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":body}]
            ).choices[0].message.content
        except Exception as e:
            print(f"OpenAI call failed (attempt {tries + 1}): {e}")
    raise RuntimeError("OpenAI call kept timing out")


def run_thread(transformations, queue):
    cn = 0
    retry_count = 0
    pbar = tqdm(total=len(transformations), desc="Generating ARC transforms")
    file_str = ""
    names = []
    previous_func = ""
    previous_test = ""
    error = None

    while cn < len(transformations):
        if retry_count > 15:
            print("❌ Too many retries, stopping.")
            retry_count = 0
            error = None
            cn += 1
            pbar.update(1)
            continue

        if (retry_count + 1) % 3 == 0:
            retry_count += 1
            time.sleep(5)  # Wait for 5 seconds before retrying
            error = None
            continue 

        elem = transformations[cn]
        name = elem["name"]
        # print(name)
        output = None
        function_code = ""
        test_code = ""

        try : 
            output = safe_openai_call(str(elem), fixer=(error is not None))
            function_code = output.split("start1")[1].split("end1")[0].strip()
            test_code = output.split("start2")[1].split("end2")[0].strip()

        except Exception as e:
            continue 


        try:

            # Execution context
            local_vars = {'np': np}

            # Define the function and the test function
            exec(function_code, globals(), local_vars)
            exec(test_code, globals(), local_vars)

            # Retrieve them from local_vars
            fn = local_vars[name]        # The transformation function
            test_fn = local_vars["test"] # The test function

            # Run the test with fn as input
            is_valid, error_str = run_with_timeout(lambda: test_fn(fn), timeout=3)
            if is_valid:
                print("✅ Test passed!")
                pbar.update(1)
                cn += 1
                error = None
                names.append(name)
                # Add the function to the file string
                file_str += function_code + "\n"
                retry_count = 0
            else :
                print(f"❌ Test failed: {error_str}")
                error = error_str
                previous_func = function_code
                previous_test = test_code
                retry_count += 1


        except Exception as e:
            print(f"❌ Error in function or test generation: {e}")
            error = str(e)
            previous_func = function_code
            previous_test = test_code
            retry_count += 1
    queue.put((file_str, names))
    return

num_threads = 8
transformations = []
len_transformations = len(function_dict.transformations)
# Divide the transformations into chunks for each thread
chunk_size = len_transformations // num_threads
# Create a list of threads
q = queue.Queue()
for i in range(num_threads):
    start_index = i * chunk_size
    end_index = (i + 1) * chunk_size if i < num_threads - 1 else len_transformations
    thread_transformations = function_dict.transformations[start_index:end_index]
    transformations.append(thread_transformations)

# Create a list to hold the threads
threads = []
# Start the threads
for i in range(num_threads):
    thread = threading.Thread(target=run_thread, args=(transformations[i], q))
    threads.append(thread)
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()
# Collect results from the queue
file_str = ""
names = []
while not q.empty():
    result = q.get()
    if isinstance(result, tuple):
        file_str += result[0]
        names.extend(result[1])
    else:
        print(f"Unexpected result in queue: {result}")
# New we will save file_str as a python file, with the added aaddition of a dict that maps from the names to the functions
with open("generated_functions.py", "w") as f:
    f.write(file_str)
    f.write("\n")
    f.write("functions = {\n")
    for name in names:
        f.write(f"    '{name}': {name},\n")
    f.write("}\n")