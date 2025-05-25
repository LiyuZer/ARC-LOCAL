import numpy as np
import scipy
import json
from tqdm import tqdm
import random
import function_dict
import generated_functions
import uuid
from data_generator import *
'''
We will define a function block, it will have a variables that are mapped to types GriD, INT, BOOL, FUNCTION, LIST_GRID.
And also a map representing from a string that is the function that is called with the variable to the next block. 

These nodes will be in a tree, we can insert a node, by adding a child node to the function block, this will lead to the next child 
inheritigng the udpated variables
'''

class function_block:
    def __init__(self, name, variables=None):
        self.name = name
        self.variables = variables if variables is not None else {}
        self.children = {}  # Maps function call to next function block

    def add_child(self, function_name, child_block):
        self.children[function_name] = child_block

    def __repr__(self):
        return f"FunctionBlock(name={self.name}, signature={self.signature}, variables={self.variables})"


'''
1: GRID→GRID, 2: GRID→INT, 3: GRID→BOOL, 4: GRID,GRID→GRID, 5: GRID,INT→GRID, 6: GRID,FUNCTION→GRID,
7: LIST_GRID→LIST_GRID, 8: LIST_GRID→GRID, 9: LIST_GRID→INT, 10: LIST_GRID,INT→LIST_GRID,
11: LIST_GRID,FUNCTION→LIST_GRID, 12: LIST_GRID,GRID→GRID, 13: GRID,FUNCTION,FUNCTION→GRID,
14: LIST_GRID,INT,INT→GRID, 15: FUNCTION,INT→FUNCTION

When we transition we use any one of these types, that is we take a variable, transform it and return a new variable.
Update the variable map with this new variable and create a new function block that has this variable map.

The key is to respect the type system, so we can only transition from one type to another if the function signature allows it.
'''

SIGNATURE_MAP = {
    1: (['GRID'], 'GRID'),
    2: (['GRID'], 'INT'),
    3: (['GRID'], 'BOOL'),
    4: (['GRID', 'GRID'], 'GRID'),
    5: (['GRID', 'INT'], 'GRID'),
    6: (['GRID', 'FUNCTION'], 'GRID'),
    7: (['LIST_GRID'], 'LIST_GRID'),
    8: (['LIST_GRID'], 'GRID'),
    9: (['LIST_GRID'], 'INT'),
    10: (['LIST_GRID', 'INT'], 'LIST_GRID'),
    11: (['LIST_GRID', 'FUNCTION'], 'LIST_GRID'),
    12: (['LIST_GRID', 'GRID'], 'GRID'),
    13: (['GRID', 'FUNCTION', 'FUNCTION'], 'GRID'),
    14: (['LIST_GRID', 'INT', 'INT'], 'GRID'),
    15: (['FUNCTION', 'INT'], 'FUNCTION'),
}

class FunctionSampler:
    def __init__(self, function_ls):
        self.signature_map = SIGNATURE_MAP
        self.type_func_map = {}
        self.func_output_type = {}
        # Build maps for both input and output signatures
        for f in function_ls:
            in_types, out_type = self.signature_map[f['signature']]
            key = tuple(in_types)
            self.type_func_map.setdefault(key, []).append(f)
            self.func_output_type[f['name']] = out_type

    def sample_function(self, input_types, output_type=None):
        """
        input_types: list, e.g. ['GRID', 'GRID'] or ['LIST_GRID']
        output_type: str or None. If set, only sample functions that output this type.
        Returns: (function dict, output_type) or (None, None)
        """
        key = tuple(input_types)
        funcs = self.type_func_map.get(key, [])
        if output_type is not None:
            # Filter to only those functions that have the required output type
            funcs = [f for f in funcs if self.func_output_type[f['name']] == output_type]
        if not funcs:
            return None, None
        f = random.choice(funcs)
        out_type = self.func_output_type[f['name']]
        return f, out_type
def pattern_fill_random(size):
    # Create a random pattern fill for a grid of given size, essentially a random walk from 0,0
    grid = np.zeros(size, dtype=int)
    x, y = 0, 0
    for i in range(size[0] * size[1]):
        grid[x, y] = random.randint(0, 9)
        # Move in a random direction
        direction = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])  # right, down, left, up
        x = (x + direction[0]) % size[0]
        y = (y + direction[1]) % size[1]
    return grid

def test_function(function):
    # Generate random input grids and check if the resulting output grids are all the sa,e
    # if so we return false
    output_grids = []
    input_grids = [ pattern_fill_random((30,30)) for _ in range(10)]

    try:
        for _ in range(0, 10):
            # Make it discreet by rounding up, and then capping at

            output_grid = function(input_grids[_])
            output_grid = np.round(output_grid).astype(int)
            # Cap the value to 0-9
            output_grid = np.clip(output_grid, 0, 9)
            output_grids.append(output_grid)

        # Check if all output grids are the same,
        if all(np.array_equal(output_grids[0], grid) for grid in output_grids):
            return False
        # If any outputs are greater than 30 in any dim then return false
        if any(grid.shape[0] > 30 or grid.shape[1] > 30 for grid in output_grids):
            return False
    except Exception as e:
        return False
    return True


class function_tree:
    def __init__(self, function_dict_map):
        self.root = function_block("root", {"input_grid": "GRID"})
        self.function_dict_map = function_dict_map
        self.function_sampler = FunctionSampler(self.function_dict_map)


    def add_function(self):
        # At each level we have a 5% chance of adding a new function block
        # when we do we return the function that we traversed
        current_block = self.root

        file_str = "def func(input_grid):\n"
        cn = 0
        found = False
        max_num = random.randint(1, 10)  # Randomly decide how many functions to add
        while True:
            if random.random() < 0.05:
                try : 
                    # pick the firs grid variable in rever
                    variable = None
                    for var, var_type in reversed(current_block.variables.items()):
                        if var_type == 'GRID':
                            variable = var
                            break
                    # randomly pick a function to apply to this variable
                    input_types = [current_block.variables[variable]]
                    function, output_type = self.function_sampler.sample_function(input_types, output_type=input_types[0])


                    # Now we remove the old variable and add the new variable, for the new block
                    variable_map = current_block.variables.copy()
                    # We will use a uuid to generate a new variable name
                    new_variable_name = f"var_{uuid.uuid4().hex[:8]}"
                    variable_map[new_variable_name] = output_type
                    # Create a new function block
                    new_block = function_block(function['name'], variable_map)
                    # Add the new block as a child of the current block
                    current_block.add_child(function['name'], new_block)
                    # Update the file string with the new variable = function call(variable)
                    file_str += f"    {new_variable_name} = generated_functions.{function['name']}({variable})\n"
                    cn += 1

                    current_block = new_block  # Move to the new block
                except Exception as e:
                    continue
            if cn > max_num:
                # loop in reverse and return the first grid variable
                output_variable = None
                for var, var_type in reversed(current_block.variables.items()):
                    if var_type == 'GRID':
                        output_variable = var
                        break
                file_str += f"    return {output_variable}\n"

                # Execution context
                local_vars = {'np': np, 'generated_functions': generated_functions}

                # Define the function and the test function
                exec(file_str, globals(), local_vars)
                # # Retrieve them from local_vars
                fn = local_vars['func']        # The transformation function
                # Test the function
                if test_function(fn):
                    found = True

                break
        return file_str, found


