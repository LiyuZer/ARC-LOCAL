import os 
import json
import random
import time
import numpy as np
from tqdm import tqdm
import openai
from openai import OpenAI
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from scipy.ndimage import median_filter
import requests
from io import BytesIO
import generated_functions
# Load image labelled Untitiled.jpeg, it is an image

ARC_COLOR_MAP = {
    (0, 0, 0): 0,         # Black
    (255, 255, 255): 1,   # White
    (255, 0, 0): 2,       # Red
    (0, 255, 0): 3,       # Green
    (0, 0, 255): 4,       # Blue
    (255, 255, 0): 5,     # Yellow
    (255, 0, 255): 6,     # Magenta
    (0, 255, 255): 7,     # Cyan
    (128, 128, 128): 8,   # Gray
    (255, 165, 0): 9,     # Orange
}
# ARC color map: index → RGB
ARC_COLOR_PALETTE = {
    0: (0, 0, 0),         # Black
    1: (255, 255, 255),   # White
    2: (255, 0, 0),       # Red
    3: (0, 255, 0),       # Green
    4: (0, 0, 255),       # Blue
    5: (255, 255, 0),     # Yellow
    6: (255, 0, 255),     # Magenta
    7: (0, 255, 255),     # Cyan
    8: (128, 128, 128),   # Gray
    9: (255, 165, 0),     # Orange
}
# Now we need to convert the image array to a grid of numbers
def discreetize(image_array):
    grid = []
    for row in image_array:
        grid_row = []
        for pixel in row:
            # Find the closest color in the color map
            closest_color = min(ARC_COLOR_MAP.keys(), key=lambda c: np.linalg.norm(np.array(c) - np.array(pixel)))
            grid_row.append(ARC_COLOR_MAP[closest_color])
        grid.append(grid_row)
    return np.array(grid)
def show_arc_grid(grid):
    """
    Given a 2D grid with integer values from 0 to 9, show the RGB image.
    """
    h, w = grid.shape
    rgb_array = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            color = ARC_COLOR_PALETTE.get(grid[i, j], (0, 0, 0))  # default to black
            rgb_array[i, j] = color

    img = Image.fromarray(rgb_array, 'RGB')
    img = img.resize((h * 10, w * 10), resample=Image.NEAREST)  # upscale for visibility
    img.show()



def arc_grid_to_rgb(grid):
    """
    Convert a 2D grid of ARC color values (0–9) into an RGB image.
    """
    h, w = grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            rgb[i, j] = ARC_COLOR_PALETTE.get(grid[i, j], (0, 0, 0))
    return rgb

def show_inputs_and_outputs(inputs, outputs):
    """
    Display RGB-rendered input-output ARC grids side by side.
    """
    fig, axs = plt.subplots(len(inputs), 2, figsize=(10, len(inputs) * 3))
    for i in range(len(inputs)):
        rgb_input = arc_grid_to_rgb(inputs[i])
        rgb_output = arc_grid_to_rgb(outputs[i])

        axs[i, 0].imshow(rgb_input)
        axs[i, 0].set_title(f'Input {i+1}')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(rgb_output)
        axs[i, 1].set_title(f'Output {i+1}')
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

def max_filter(grid, kernel_size=3):
    """
    Apply a max filter to the grid with a given kernel size.
    """
    return maximum_filter(grid, size=kernel_size)
def med_filter(grid, kernel_size=3):
    """
    Apply a median filter to the grid with a given kernel size.
    """
    return median_filter(grid, size=kernel_size)


def convert_image_to_grid(image, size=(30, 30), filter_type='med', kernel_size=3):
    image = image.resize(size)
    # Now we need to convert the image to a numpy array
    image_array = np.array(image)

    # normalize thje image to a mean of 0 and std of 1
    image_array = (image_array - np.mean(image_array)) / (np.std(image_array))
    image_array = image_array * 255
    grid = discreetize(image_array)
    if filter_type == 'max':
        # Apply max filter
        grid = max_filter(grid, kernel_size=kernel_size)
    elif filter_type == 'med':
        # Convert the image to a grid
        grid = med_filter(grid, kernel_size=1)
    return grid



# We will have a Dataset class, that takes in the image urls, and when queried, can give us the images scaled to the required size 

class DatasetImages:
    def __init__(self, image_urls):
        self.image_urls = image_urls

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return convert_image_to_grid(self.images[idx])

    def load_random_images(self, sizes, filter_types, kernel_sizes, n=10):
        """
        Load n random images from the image urls
        """
        cn = 0 
        grids = []
        while cn < n:
            # Get a random image url
            image_url = random.choice(self.image_urls)
            # print(f"Loading image {cn+1}/{n} from {image_url}")
            # Get the image
            response = requests.get(image_url, stream=True, timeout=10)
            image = None
            if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                # print("Invalid response or non-image content")
                continue
            image = Image.open(BytesIO(response.content))
            # Convert the image to a grid
            grid = convert_image_to_grid(image, sizes[cn], filter_types[cn], kernel_sizes[cn])
            # Append the grid to the list
            grids.append(grid)
            cn += 1
        return grids


# Now we get to the pattern_generation part
'''
A patern generator class takes in a setup, and then uses the setup to output training testing examples. 
Note the class has a dataset, that it will use to both get objects and inputs 
setup = {
    'num_total_example_pairs': int,            # e.g., 8
    'num_objects': int,                        # e.g., 12
    'sizes_input': List[Tuple[int, int]],      # input grid size per example
    'sizes_output': List[Tuple[int, int]],     # output grid size per example
    'objects_sizes': List[Tuple[int, int]],    # each object's shape
    'num_testing_examples': int                # ⩽ num_total_example_pairs
}
'''
class PatternGenerator:
    def __init__(self, dataset):
        self.dataset = dataset

    def generate(self, setup, function):
        # We will use the setup to generate the examples
        # The return will be a dict as follows 
        # { 'train': [ {'input' : [], 'output': []},..  ], 'test': [same as train] }
        # We will use the dataset to get the objects and inputs

        # Generate objects 
        # Kernel size will always be 1/6 of the max_dim
        kernel_sizes = [10 for s in setup['objects_sizes']]
        objects = self.dataset.load_random_images(setup['objects_sizes'], ['med'] * setup['num_objects'], kernel_sizes, n=setup['num_objects'])
        # Generate inputs
        kernel_sizes = [10 for s in setup['sizes_input']]
        inputs = self.dataset.load_random_images(setup['sizes_input'], ['med'] * setup['num_total_example_pairs'], kernel_sizes, n=setup['num_total_example_pairs'])
        id_m = 0
        # Now we need to generate the outputs
        outputs = []
        real_inputs = []
        for i in range(setup['num_total_example_pairs']):
            # Get the input
            input_grid = inputs[i]
            output = function(input_grid)
            # Append the output to the list
            outputs.append(output)
            id_m += 1
        return inputs, outputs


