import random
import json
from datasets import Dataset

# Load ARC data
with open('arc-agi_training_challenges.json') as f:
    train = json.load(f)
with open('arc-agi_training_solutions.json') as f:
    test = json.load(f)

examples = []
for task_id, task in train.items():
    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    # Use corresponding outputs from solutions file (if needed)
    test_outputs = test.get(task_id, [])
    if not train_pairs or not test_pairs or not test_outputs:
        continue  # Skip incomplete tasks

    # Loop over available test inputs/outputs
    for i, test_pair in enumerate(test_pairs):
        test_input = test_pair["input"]
        # The completion is the output for that test input
        if i >= len(test_outputs):
            continue  # Skip if not aligned
        test_output = test_outputs[i]

        # Build input/output interleaved sequence
        flat_io = []
        cn = 0
        for pair in train_pairs:
            flat_io.append("input" + str(cn))
            flat_io.append(str(pair["input"]))
            flat_io.append("output" + str(cn))
            flat_io.append(str(pair["output"]))
            cn += 1
        flat_io.append("input" + str(cn))
        flat_io.append(str(test_input))  # Add test input at the end
        flat_io = " ".join(flat_io)
        print(flat_io)
        # Add to dataset
        examples.append({
            "inputs_outputs": flat_io,   # This is the input to the model
            "completion": str(test_output)    # This is what the model should predict
        })

# Convert to HuggingFace Dataset
hf_dataset = Dataset.from_list(examples)
hf_dataset.save_to_disk("arc_agi_final_hf_dataset")

print(hf_dataset[0])
