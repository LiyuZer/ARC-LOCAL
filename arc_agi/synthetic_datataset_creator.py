from transformation_generator import *
from datasets import load_from_disk
import threading
# #Let us test this out 
# function_e = function_tree(function_dict.transformations)
# found_ls = []
# pbar = tqdm(total=10000, desc="Generating functions")
# # Gather 1000 functions
# while len(found_ls) < 10000:
#     # Add a function to the tree
#     file_str, found = function_e.add_function()
#     if found:
#         found_ls.append(file_str)
#         pbar.update(1)

    

# # Open the file open-fimages-dataset-train0.tsv and read the first 1000 lines
# lines = []
# with open('open-images-dataset-train0.tsv', 'r') as f:
#     for line in f:
#         lines.append(line)
# # Now we have the lines in a list
# lines = lines[1:100000]

# # Each line has two tabs separated values, the first is the image id and the second is the image url
# image_urls = []
# for line in lines:
#     image_urls.append(line.split('\t')[0])


# # let us just test the pattern generator
# dataset = DatasetImages(image_urls)
# pattern_generator = PatternGenerator(dataset)


# def get_example(func_ls , ls):
#     final_ls = []
#     pbar = tqdm(total=len(func_ls), desc="Generating examples")
#     for func in func_ls:
#         try:
#             num_examples = random.randint(1, 8)
#             setup = {
#                 'num_total_example_pairs': num_examples,
#                 'num_objects': 1,
#                 'sizes_input': [(random.randint(1,30), random.randint(1,30))] * num_examples,     # We ignore this since base_input is overwritten
#                 'sizes_output': [(10, 10)] * 5,  # Final normalized output size
#                 'objects_sizes': [(5, 5)],
#                 'num_testing_examples': 2
#             }

#             # Execution context
#             local_vars = {'np': np, 'generated_functions': generated_functions}

#             # Define the function and the test function
#             exec(func, globals(), local_vars)

#             fn = local_vars['func'] 

#             inputs, outputs = pattern_generator.generate(setup, fn)
#             input_output_ls = []
#             for i in range(len(inputs) - 1):
#                 input_output_ls.append(f"input{i}")
#                 input_output_ls.append(str(inputs[i]))
#                 input_output_ls.append(f"output{i}")
#                 input_output_ls.append(str(outputs[i]))
#             input_output_ls.append(f"input{len(inputs)-1}")
#             input_output_ls.append(str(inputs[-1]))
#             # join 
#             input_output_str = " ".join(input_output_ls)
#             final_ls.append({
#                 "inputs_outputs": input_output_str,   # This is the input to the model
#                 "completion": str(outputs[-1])         # This is what the model should predict
#             })
#             pbar.update(1)
#         except Exception as e:
#             pbar.update(1)
#             continue
#     ls.extend(final_ls)
#     return ls
# ls = []
# # We need around 10,000 training examples
# num_training_examples = 10000
# # We will use threading to speed up the process
# threads = []
# div = len(found_ls) // 10  # Divide the found_ls into 10 parts for 10 threads
# for i in range(10):  # 10 threads
#     start_index = i * div
#     end_index = (i + 1) * div if i < 9 else len(found_ls)  # Last thread takes the rest
#     thread_found_ls = found_ls[start_index:end_index]
#     thread = threading.Thread(target=get_example, args=(thread_found_ls, ls))
#     threads.append(thread)
# # Start the threads
# for thread in threads:
#     thread.start()
# # Wait for all threads to finish
# for thread in threads:
#     thread.join()


# print(len(ls))  
# # Save the dataset to disk
# hf_dataset = Dataset.from_list(ls)
# hf_dataset.save_to_disk("synthetic_function_dataset")
