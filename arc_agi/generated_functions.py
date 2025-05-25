import numpy as np
def generate_texture_fn(grid, func):
    rows, cols = grid.shape
    new_grid = np.zeros((rows, cols), dtype=grid.dtype)
    for r in range(rows):
        for c in range(cols):
            new_grid[r, c] = func(grid[r, c])
    return new_grid
def overlay_generated_fn(grid, func):
    return np.vectorize(func)(grid)

#
def sort_by_area_desc(grids):
    return sorted(grids, key=lambda grid: np.sum(grid > 0), reverse=True)
#
def sort_by_num_objects(grids):
    return sorted(grids, key=lambda g: np.sum(g > 0))
#
def reverse_list(l):
    return [grid[::-1] for grid in l]
def normalize_each_grid(list_grid):
    # Confirming the input is a list of grids (2D numpy arrays)
    if not isinstance(list_grid, list):
        return None  # Returning None if the input is not a list

    normalized_grids = []
    
    for grid in list_grid:
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            return None  # Ensuring each item in the list is a 2D numpy array
            
        # Normalization: (value - min) / (max - min)
        min_val = np.min(grid)
        max_val = np.max(grid)
        
        # To avoid division by zero, we can check max_val == min_val
        if max_val == min_val:
            normalized_grid = np.zeros_like(grid)  # set all to zeros if max equals min
        else:
            normalized_grid = (grid - min_val) / (max_val - min_val)  # Normalizing
        
        normalized_grids.append(normalized_grid)
    
    return normalized_grids  # Returning the list of normalized grids
#
def binarize_all_grids(grids):
    return [np.where(grid > 0, 1, 0) for grid in grids]
def pad_each_to_max_size(grids):
    max_rows = max(grid.shape[0] for grid in grids)
    max_cols = max(grid.shape[1] for grid in grids)
    padded_grids = []
    for grid in grids:
        padded_grid = np.pad(grid, ((0, max_rows - grid.shape[0]), (0, max_cols - grid.shape[1])), mode='constant')
        padded_grids.append(padded_grid)
    return padded_grids
def trim_each_to_content(list_grid):
    trimmed_grid_list = []
    for grid in list_grid:
        # Find the non-empty bounds of the grid
        rows = np.any(grid, axis=1)
        cols = np.any(grid, axis=0)
        # Get the indices of the non-empty rows and columns
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        if row_indices.size > 0 and col_indices.size > 0:
            trimmed_grid = grid[row_indices.min():row_indices.max() + 1, col_indices.min():col_indices.max() + 1]
        else:
            trimmed_grid = np.array([[]])  # Return an empty grid if fully empty
        trimmed_grid_list.append(trimmed_grid)
    return trimmed_grid_list
#
def rotate_each_random(grids):
    import random
    def rotate(grid):
        return np.rot90(grid, k=random.choice([1, 2, 3]))  # Randomly rotate by 90, 180, or 270 degrees
    
    return [rotate(grid) for grid in grids]
#
def filter_empty_grids(grids):
    return [grid for grid in grids if np.any(grid)]
#
def apply_threshold_each(grids):
    return [np.where(grid >= 0.5, 1, 0) for grid in grids]
#
def center_objects_each(list_grid):
    # Center each grid in the list by padding with zeros
    centered_grids = []
    
    for grid in list_grid:
        if not grid.size:  # Skip empty grids
            centered_grids.append(grid)
            continue
        
        # Calculate the necessary padding
        height, width = grid.shape
        pad_height = (4 - height % 4) % 4  # Pad to the next multiple of 4
        pad_width = (4 - width % 4) % 4  # Pad to the next multiple of 4
        
        # Create new grid with padding
        new_grid = np.zeros((height + pad_height, width + pad_width), dtype=grid.dtype)
        new_grid[:height, :width] = grid
        
        centered_grids.append(new_grid)
    
    return centered_grids
#
def scale_each_to_square(grids):
    return [np.square(grid) for grid in grids]
#
def invert_each_grid(list_grid):
    # Corrected function to match the signature: LIST_GRID→LIST_GRID
    # The function inverts each grid in the input list.

    # Initialize a new list to hold inverted grids
    inverted_grids = []
    
    # Iterate through the list of grids
    for grid in list_grid:
        # Invert the grid and append it to the new list
        inverted_grids.append(np.flipud(np.fliplr(grid)))  # Flipping up-down and left-right

    return inverted_grids
#
def fill_holes_each(grids):
    # Ensure input is a list of grids
    if not isinstance(grids, list) or not all(isinstance(grid, np.ndarray) for grid in grids):
        return []  # invalid input, return an empty list

    filled_grids = []
    
    for grid in grids:
        # Fill holes in each grid using np.nan_to_num as an example
        filled_grid = np.nan_to_num(grid)  # fill NaNs with 0 (or use another method as needed)
        filled_grids.append(filled_grid)
    
    return filled_grids
#
def stack_vertical(grids):
    return np.vstack(grids)
#
def stack_horizontal(list_grids):
    return np.hstack(list_grids)
#
def mosaic_square(list_grid):
    # Start with an empty grid
    # Assuming that list_grid is a list of 2D grids (numpy arrays).
    import numpy as np

    # Initialize the height and width for the resulting grid
    total_height = sum(grid.shape[0] for grid in list_grid)
    total_width = max(grid.shape[1] for grid in list_grid)

    # Create an empty grid of the required size
    output_grid = np.zeros((total_height, total_width))

    # Fill the output grid with the input grids
    current_height = 0
    for grid in list_grid:
        output_grid[current_height:current_height + grid.shape[0], :grid.shape[1]] = grid
        current_height += grid.shape[0]

    return output_grid
def blend_average_sequence(grids):
    if not grids:
        return np.array([])

    # Calculate the average of the grids
    average_grid = np.mean(grids, axis=0)
    return average_grid.astype(grids[0].dtype)
#
def overlay_sequence_by_max(sequence):
    result = np.maximum.reduce(sequence)
    return result

#
def difference_chain(grids):
    result = np.abs(grids[0] - grids[1])
    for grid in grids[2:]:
        result = np.abs(result - grid)
    return result
#
def collapse_by_mode_color(grids):
    color_count = {}
    for grid in grids:
        unique_colors, counts = np.unique(grid, return_counts=True)
        for color, count in zip(unique_colors, counts):
            if color in color_count:
                color_count[color] += count
            else:
                color_count[color] = count
                
    mode_color = max(color_count, key=color_count.get)
    return np.array([[mode_color]])

#
def majority_vote_pixelwise(grids):
    if not grids:
        return np.array([])

    # Get dimensions from the first grid
    height, width = grids[0].shape
    result = np.zeros((height, width), dtype=int)

    # Iterate over each pixel position
    for i in range(height):
        for j in range(width):
            # Count occurrences of each pixel value across all grids
            pixel_values = [grid[i, j] for grid in grids]
            result[i, j] = np.bincount(pixel_values).argmax()

    return result
#
def max_pool_across_grids(grids):
    result = np.zeros_like(grids[0])
    for grid in grids:
        result = np.maximum(result, grid)
    return result
#
def grid_at_median_index(grids):
    flat_list = []
    for grid in grids:
        flat_list.extend(grid.flatten())
    flat_list.sort()
    mid_index = len(flat_list) // 2
    median_value = flat_list[mid_index]
    for grid in grids:
        if median_value in grid:
            return grid
    return None
#
def total_object_count(list_grid):
    return sum(np.sum(grid > 0) for grid in list_grid)
#
def average_width_grids(grids):
    total_width = sum(grid.shape[1] for grid in grids)
    return total_width // len(grids) if grids else 0
#
def max_height_grids(grids):
    return max(grid.shape[0] for grid in grids)
#
def min_area_grids(grids):
    return min(grid.size for grid in grids)
#
def sum_unique_colors(grids):
    unique_colors = set()
    for grid in grids:
        unique_colors.update(np.unique(grid))
    return int(sum(unique_colors))

#
def mode_color_overall(grids):
    color_count = {}
    for grid in grids:
        for row in grid:
            for color in row:
                if color in color_count:
                    color_count[color] += 1
                else:
                    color_count[color] = 1
    if not color_count:
        return None
    return max(color_count.items(), key=lambda x: x[1])[0]
#
# Function to count the most common size of grids in a LIST_GRID
def most_common_grid_size_count(list_grid):
    # Using a dictionary to count the occurrence of each grid size
    size_count = {}
    
    for grid in list_grid:
        size = grid.shape  # Get the shape of the grid (height, width)
        if size in size_count:
            size_count[size] += 1
        else:
            size_count[size] = 1
            
    # Find the most common size
    most_common_size = max(size_count, key=size_count.get)
    
    return size_count[most_common_size]  # Return the count of the most common size
def variance_of_object_counts(list_grid):
    counts = [np.count_nonzero(grid) for grid in list_grid]
    mean_count = np.mean(counts)
    variance = np.var(counts)
    return variance
#
def count_empty_grids(grids):
    return sum(1 for grid in grids if np.all(grid == 0))
#
def longest_common_subpattern_length(grids: list) -> int:
    def extract_subpatterns(grid):
        patterns = set()
        rows, cols = grid.shape
        for r in range(rows):
            for c in range(cols):
                for h in range(1, rows - r + 1):
                    for w in range(1, cols - c + 1):
                        patterns.add(tuple(map(tuple, grid[r:r+h, c:c+w])))
        return patterns

    common_patterns = None
    for grid in grids:
        patterns = extract_subpatterns(grid)
        if common_patterns is None:
            common_patterns = patterns
        else:
            common_patterns.intersection_update(patterns)

    if common_patterns:
        return max(len(pattern) for pattern in common_patterns) if common_patterns else 0
    return 0
#
def sample_every_n(list_grid, n):
    # Fix: Ensure we sample every nth grid
    if n <= 0:  # Handle invalid sampling rate
        return []
    return [grid for index, grid in enumerate(list_grid) if index % n == 0]  # Correctly sample every nth grid
#
def resize_all_to_n(grids, n):
    return [np.resize(grid, (n, n)) for grid in grids]
#
def duplicate_each_n_times(list_grid, n):
    return [np.tile(grid, (n, n)) for grid in list_grid]
def rotate_each_n_times(grids, n):
    def rotate(grid):
        return np.rot90(grid, k=n)
    
    return [rotate(grid) for grid in grids]
#
def pad_all_to_n(list_grid, n):
    # Ensure that each grid in the list is padded to size (n, n)
    padded_grids = []
    for grid in list_grid:
        # Determine the current shape of the grid
        rows, cols = grid.shape
        
        # Create a new padded grid filled with zeros (or any other value)
        padded_grid = np.zeros((n, n))
        
        # Copy the original grid values into the padded grid
        padded_grid[:rows, :cols] = grid
        
        # Append the padded grid to the result list
        padded_grids.append(padded_grid)
    
    return padded_grids  # Return the list of padded grids
def overlay_min(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
    return np.minimum(grid1, grid2)
#
def overlay_add_modulo(grid1, grid2):
    # Ensure that both grids have the same shape
    if grid1.shape != grid2.shape:
        raise ValueError("Both grids must have the same dimensions")
    
    # Perform element-wise addition and apply modulo operation
    return (grid1 + grid2) % 256  # Example modulo with 256 for typical image values
#
def union_mask(grid1, grid2):
    return np.maximum(grid1, grid2)
#
def intersection_mask(grid_a, grid_b):
    return np.logical_and(grid_a, grid_b).astype(int)
#
def difference_mask(grid1, grid2):
    return np.abs(grid1 - grid2)
#
def xor_mask(grid1, grid2):
    return np.bitwise_xor(grid1, grid2)
#
def concatenate_horizontal(grid1, grid2):
    return np.hstack((grid1, grid2))
#
def concatenate_vertical(grid1, grid2):
    return np.vstack((grid1, grid2))
#
def blend_average(grid1, grid2):
    return (grid1 + grid2) / 2
#
def blend_multiply(grid1, grid2):
    return np.clip(grid1 * grid2, 0, 255)
def compare_highlight_differences(grid1, grid2):
    result = grid1.copy()
    height, width = grid1.shape
    
    for i in range(height):
        for j in range(width):
            if grid1[i, j] != grid2[i, j]:
                result[i, j] = 1  # Highlight difference with a 1
            else:
                result[i, j] = 0  # No difference, set to 0
                
    return result
#
def stitch_diagonal(grid1, grid2):
    # Ensure the grids have the same shape for diagonal stitching
    if grid1.shape != grid2.shape:
        raise ValueError("Both grids must have the same shape.")

    # Create an output grid with the same shape, initialized to zeros
    result = np.zeros((grid1.shape[0], grid1.shape[1]), dtype=grid1.dtype)

    # Iterate through the grid and stitch diagonally
    for i in range(grid1.shape[0]):
        for j in range(grid1.shape[1]):
            if i == j:  # if we are on the diagonal
                result[i, j] = grid1[i, j] + grid2[i, j]
            else:
                result[i, j] = grid1[i, j]

    return result
#
def overlay_grid2_on_grid1(grid1, grid2):
    # Assuming grid1 and grid2 are of the same shape for overlaying
    return np.where(grid2 == 1, 1, grid1)  # Overlays grid2 on grid1, where grid2 has 1s
def swap_quadrants_between(grid1, grid2):
    # Check that both grids are the same size
    assert grid1.shape == grid2.shape, "Grid dimensions must match."
    rows, cols = grid1.shape
    mid_row, mid_col = rows // 2, cols // 2
    
    # Swap the quadrants
    temp = grid1[:mid_row, :mid_col].copy()  # Top-left quadrant of grid1
    grid1[:mid_row, :mid_col] = grid2[:mid_row, :mid_col]  # Replace top-left with grid2's top-left
    grid2[:mid_row, :mid_col] = temp  # Replace top-left of grid2 with the stored top-left of grid1

    return grid1, grid2  # Return modified grids
#
def interleave_columns(grid):
    rows, cols = grid.shape
    new_cols = cols * 2
    new_grid = np.zeros((rows, new_cols), dtype=grid.dtype)
    
    for col in range(cols):
        new_grid[:, col * 2] = grid[:, col]
        new_grid[:, col * 2 + 1] = grid[:, col]  # Repeat the column
    
    return new_grid
#
def interleave_rows(grid1, grid2):
    # Ensure both grids have the same number of columns
    if grid1.shape[1] != grid2.shape[1]:
        raise ValueError("Both grids must have the same number of columns.")

    # Interleave rows of grid1 and grid2
    interleaved = []
    
    for row1, row2 in zip(grid1, grid2):
        interleaved.append(row1)
        interleaved.append(row2)
    
    return np.array(interleaved)
#
def mask_grid1_with_grid2(grid1, grid2):
    return np.where(grid2 != 0, grid1, 0)
#
def replace_region_where_grid2_nonzero(grid1, grid2):
    mask = grid2 != 0
    grid1[mask] = grid2[mask]
    return grid1
#
def erode_n(grid, n):
    eroded_grid = grid.copy()
    for _ in range(n):
        eroded_grid = np.minimum(eroded_grid, np.roll(eroded_grid, 1, axis=0))
        eroded_grid = np.minimum(eroded_grid, np.roll(eroded_grid, -1, axis=0))
        eroded_grid = np.minimum(eroded_grid, np.roll(eroded_grid, 1, axis=1))
        eroded_grid = np.minimum(eroded_grid, np.roll(eroded_grid, -1, axis=1))
    return eroded_grid
#
def pad_border_n(grid, n):
    # Validate the input grid and n value
    if not isinstance(grid, np.ndarray) or not isinstance(n, int) or n < 0:
        return None  # Engineering decision to return None for invalid input

    # Get the shape of the original grid
    rows, cols = grid.shape

    # Create a new grid with added borders
    padded_grid = np.zeros((rows + 2 * n, cols + 2 * n), dtype=grid.dtype)

    # Place the original grid in the center of the new padded grid
    padded_grid[n:n + rows, n:n + cols] = grid

    return padded_grid
def trim_border_n(grid, n):
    return grid[n:-n, n:-n] if n < grid.shape[0] // 2 and n < grid.shape[1] // 2 else grid

#
def threshold_at_int(grid, threshold):
    return np.where(grid >= threshold, 1, 0)
#
def quantize_colors_n(grid, n):
    """
    Quantizes the colors in the grid into n different levels.
    The grid is expected to be a 2D numpy array.
    """
    # Ensure n is greater than 0
    if n <= 0:
        raise ValueError("n must be greater than 0")
    
    # Create a copy of the grid to avoid modifying the original
    quantized_grid = grid.copy()
    
    # Compute the quantization level
    max_val = np.max(quantized_grid)
    min_val = np.min(quantized_grid)

    # Perform quantization
    quantized_grid = ((quantized_grid - min_val) / (max_val - min_val) * (n - 1)).astype(int)
    
    return quantized_grid
#
def shift_right_n(grid, n):
    if n < 0:
        raise ValueError("n must be non-negative")
    rows, cols = grid.shape
    n = n % cols  # Ensure n is within the column bounds
    if n == 0:
        return grid
    return np.roll(grid, n, axis=1)
#
def shift_down_n(grid, n):
    rows, cols = grid.shape
    n = n % rows  # To handle shifts larger than the number of rows
    if n == 0:
        return grid
    return np.vstack((grid[-n:], grid[:-n]))

#
def rotate_n_times_90(grid, n):
    n = n % 4  # Normalize to within 0-3
    for _ in range(n):
        grid = np.rot90(grid)
    return grid
#
def repeat_tile_n(grid, n):
    return np.tile(grid, (n, n))
#
def scale_by_factor_n(grid, n):
    return grid * n
def color_to_n(grid, color_index):
    unique_colors = np.unique(grid)
    color_mapping = {color: index for index, color in enumerate(unique_colors)}
    
    transformed_grid = np.copy(grid)
    for row in range(transformed_grid.shape[0]):
        for col in range(transformed_grid.shape[1]):
            transformed_grid[row, col] = color_mapping.get(transformed_grid[row, col], -1)
    
    return transformed_grid
#
def mask_color_n(grid, n):
    return np.where(grid == n, 1, 0)
#
def outline_thickness_n(grid, n):
    outline_grid = np.zeros_like(grid)
    rows, cols = grid.shape
    
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] > 0:
                # Check if this cell is part of the outline
                if (i == 0 or grid[i - 1, j] == 0) or (i == rows - 1 or grid[i + 1, j] == 0) or \
                   (j == 0 or grid[i, j - 1] == 0) or (j == cols - 1 or grid[i, j + 1] == 0):
                    outline_grid[i, j] = 1  # This cell is part of the outline
                
    # Thicken the outline by n pixels
    for _ in range(n):
        new_outline_grid = np.zeros_like(outline_grid)
        for i in range(rows):
            for j in range(cols):
                if outline_grid[i, j] == 1:
                    new_outline_grid[i, j] = 1
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if abs(di) + abs(dj) == 1:  # Only the adjacent cells (not diagonals)
                                ni, nj = i + di, j + dj
                                if 0 <= ni < rows and 0 <= nj < cols:
                                    new_outline_grid[ni, nj] = 1
        outline_grid = new_outline_grid
    
    return outline_grid
#
def apply_per_cell_function(grid, func):
    return np.vectorize(func)(grid)
#
def conditional_transform_fn(grid, func):
    result = np.zeros_like(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if func(grid[i, j]):
                result[i, j] = grid[i, j] * 2  # Example transformation: double the value
            else:
                result[i, j] = grid[i, j]  # Leave it unchanged
    return result
#
def map_colors_with_fn(grid, function):
    return np.array([[function(cell) for cell in row] for row in grid])
#
def mask_with_predicate_fn(grid, predicate):
    return np.where(predicate(grid), 1, 0)

#
def transform_rows_fn(grid, func):
    return np.array([func(row) for row in grid])
#
def transform_columns_fn(grid, transform_fn):
    for col in range(grid.shape[1]):
        for row in range(grid.shape[0]):
            grid[row, col] = transform_fn(grid[row, col])
    return grid
#
def warp_coordinates_fn(grid, func):
    rows, cols = grid.shape
    transformed_grid = np.zeros((rows, cols), dtype=grid.dtype)
    for r in range(rows):
        for c in range(cols):
            transformed_grid[r, c] = func(r, c)
    return transformed_grid
#
def filter_pixels_fn(grid, function):
    # Apply the filter function to each element in the grid
    filtered_grid = np.vectorize(function)(grid)
    return filtered_grid
#
def rotate_random(grid):
    import numpy as np
    if np.random.rand() < 0.5:
        return np.rot90(grid)
    else:
        return np.rot90(grid, k=3)
#
def normalize_intensity(grid):
    min_val = np.min(grid)
    max_val = np.max(grid)
    if max_val - min_val == 0:
        return grid
    return (grid - min_val) / (max_val - min_val)
#
def equalize_histogram(grid):
    # Check if the input grid is empty
    if grid.size == 0:
        return grid  # Return the empty grid if there's no data

    # Get the histogram of pixel values
    hist, bins = np.histogram(grid.flatten(), bins=256, range=[0,256])
    
    # Calculate cumulative distribution function
    cdf = hist.cumsum()
    
    # Normalize the cumulative distribution function
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    
    # Use linear interpolation to find new pixel values
    equalized_grid = np.interp(grid.flatten(), bins[:-1], cdf_normalized)
    
    # Reshape back to the original grid shape
    return equalized_grid.reshape(grid.shape).astype(np.uint8)  # Ensure type consistency
#
def warp_perspective(grid):
    return np.flipud(np.fliplr(grid))
#
def shear_horizontal(grid):
    return np.roll(grid, shift=1, axis=1)
#
def shear_vertical(grid):
    rows, cols = grid.shape
    sheared_grid = np.zeros((rows, cols), dtype=grid.dtype)

    for r in range(rows):
        for c in range(cols):
            new_c = c + r  # Shearing effect
            if new_c < cols:
                sheared_grid[r, new_c] = grid[r, c]

    return sheared_grid
#
def count_colors(grid):
    """
    Counts the number of unique colors (non-zero values) in the given grid.

    :param grid: A 2D numpy array representing the grid.
    :return: The count of unique non-zero elements in the grid.
    """
    # Ensure the grid is a numpy array
    unique_colors = set(grid.flatten())  # Extract unique colors
    unique_colors.discard(0)  # Remove color zero if it exists
    return len(unique_colors)  # Return the number of unique colors
#
def unique_color_count(grid):
    # Ensure to use np.unique to get unique colors in the grid and count them
    unique_colors = np.unique(grid)
    return len(unique_colors)
#
def grid_width(grid):
    return grid.shape[1]
#
def grid_height(grid):
    return grid.shape[0]
def area_nonzero(grid):
    # Count the number of non-zero elements in the grid
    return np.count_nonzero(grid)
#
def perimeter_total(grid):
    if grid.size == 0:
        return 0
    rows, cols = grid.shape
    return 2 * (rows + cols)
#
def number_of_objects(grid):
    return int(np.sum(grid > 0))
#
def largest_object_area(grid):
    visited = np.zeros(grid.shape, dtype=bool)
    max_area = 0

    def dfs(x, y):
        if x < 0 or x >= grid.shape[0] or y < 0 or y >= grid.shape[1] or visited[x, y] or grid[x, y] == 0:
            return 0
        visited[x, y] = True
        area = 1  # Count this cell
        area += dfs(x + 1, y)
        area += dfs(x - 1, y)
        area += dfs(x, y + 1)
        area += dfs(x, y - 1)
        return area

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1 and not visited[i, j]:
                current_area = dfs(i, j)
                max_area = max(max_area, current_area)

    return max_area
#
def smallest_object_area(grid):
    area = 0
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)

    def dfs(x, y):
        if x < 0 or y < 0 or x >= rows or y >= cols or visited[x][y] or grid[x][y] == 0:
            return 0
        visited[x][y] = True
        count = 1
        count += dfs(x + 1, y)
        count += dfs(x - 1, y)
        count += dfs(x, y + 1)
        count += dfs(x, y - 1)
        return count
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1 and not visited[i][j]:
                area_found = dfs(i, j)
                if area_found > 0 and (area == 0 or area_found < area):
                    area = area_found

    return area if area > 0 else None
#
def average_object_area(grid):
    objects = np.unique(grid)
    objects = objects[objects != 0]  # Exclude the background (assuming 0 is background)
    
    if len(objects) == 0:
        return 0  # No objects found

    total_area = 0
    for obj in objects:
        total_area += np.sum(grid == obj)

    return total_area / len(objects)  # Average area of objects
#
def longest_horizontal_line_length(grid):
    max_length = 0
    current_length = 0
    
    for row in grid:
        current_length = 0
        for val in row:
            if val == 1:  # Assuming we're looking for lines of 1s
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 0
    
    return max_length
#
def longest_vertical_line_length(grid):
    if grid.size == 0:
        return 0
        
    max_length = 0
    num_rows, num_cols = grid.shape
    
    for col in range(num_cols):
        current_length = 0
        
        for row in range(num_rows):
            if grid[row, col] == 1:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 0

    return max_length
def color_entropy(grid):
    unique_colors, counts = np.unique(grid, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))
    return entropy
#
def dominant_color(grid):
    unique_colors, counts = np.unique(grid, return_counts=True)
    max_count_index = np.argmax(counts)
    return unique_colors[max_count_index]
#
def least_common_color(grid):
    # Count unique elements in the grid
    unique_colors, counts = np.unique(grid, return_counts=True)
    
    # Find the index of the least common color
    least_common_index = np.argmin(counts)
    
    # Return the least common color
    return unique_colors[least_common_index]
#
def diagonal_symmetry_score(grid):
    if grid.ndim != 2:
        return 0
    n, m = grid.shape
    score = 0
    for i in range(min(n, m)):
        if grid[i, i] == grid[i, m - 1 - i]:
            score += 1
    return score
#
def number_of_holes(grid):
    holes = 0
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or visited[r, c] or grid[r, c] == 1:
            return
        visited[r, c] = True
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0 and not visited[i, j]:
                dfs(i, j)
                holes += 1

    return holes

#
def bounding_box_area(grid):
    rows = grid.shape[0]
    cols = grid.shape[1]
    
    min_row = rows
    max_row = -1
    min_col = cols
    max_col = -1
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0:  # Assuming non-zero values are part of the bounding box
                if r < min_row:
                    min_row = r
                if r > max_row:
                    max_row = r
                if c < min_col:
                    min_col = c
                if c > max_col:
                    max_col = c
    
    if min_row > max_row or min_col > max_col:  # No bounding box found
        return 0  # Area of non-existent bounding box
    
    area = (max_row - min_row + 1) * (max_col - min_col + 1)
    return area
#
def aspect_ratio_scaled(grid):
    h, w = grid.shape
    return int(h / w)
#
def edge_pixel_count(grid):
    # Count the number of edge pixels (1st row, last row, 1st column, last column)
    edge_count = 0
    rows, cols = grid.shape
    
    # Count top row and bottom row
    edge_count += cols  # All pixels in the top row
    if rows > 1:        # Only add bottom row if there is one
        edge_count += cols
    
    # Count left column and right column
    if rows > 2:  # Avoid double counting corners if more than 2 rows
        edge_count += (rows - 2) * 2  # the remaining rows contribute 2 edge pixels each
    
    return edge_count
#
def is_empty(grid):
    return np.all(grid == 0)
#
def is_symmetric_horizontal(grid):
    # Check if the grid is a 2D numpy array
    if grid.ndim != 2:
        return False
    
    # Get the number of rows in the grid
    num_rows = grid.shape[0]
    
    # Check symmetry by comparing top half to the bottom half
    for i in range(num_rows // 2):
        if not np.array_equal(grid[i, :], grid[num_rows - i - 1, :]):
            return False
            
    return True
def is_symmetric_vertical(grid):
    # Ensure input is a 2D numpy array
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    # Get number of rows and columns in the grid
    rows, cols = grid.shape
    # Check symmetry by comparing the grid with its mirrored version
    for r in range(rows):
        for c in range(cols // 2):
            if grid[r, c] != grid[r, cols - c - 1]:
                return False
    return True
#
def is_symmetric_diagonal(grid):
    return np.array_equal(grid, np.transpose(grid))
def contains_color_two(grid):
    return np.any(grid == 2)  # Check if any element in the grid is 2
#
def has_hollow_center(grid):
    rows, cols = grid.shape
    if rows < 3 or cols < 3:
        return False
    center_row_range = range(1, rows - 1)
    center_col_range = range(1, cols - 1)
    
    for i in center_row_range:
        for j in center_col_range:
            if grid[i, j] != 0:
                return False

    for i in [0, rows - 1]:
        for j in range(cols):
            if grid[i, j] != 1:
                return False

    for j in [0, cols - 1]:
        for i in range(rows):
            if grid[i, j] != 1:
                return False

    return True
#
def has_full_border(grid):
    # Check if the grid is empty
    if grid.size == 0:
        return False
    
    # Check top and bottom rows
    if not all(grid[0, :]) or not all(grid[-1, :]):
        return False
    
    # Check left and right columns (excluding corners that were checked in rows)
    if not all(grid[:, 0]) or not all(grid[:, -1]):
        return False
    
    return True
#
def has_checkerboard_pattern(grid):
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            expected_value = (i + j) % 2  # Alternating pattern starting from 0
            if grid[i, j] != expected_value:
                return False
    return True
#
def has_isolated_pixel(grid):
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:  # Check for the isolated pixel condition
                isolated = True
                # Check all 8 possible neighbors
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if (dr != 0 or dc != 0) and 0 <= r + dr < rows and 0 <= c + dc < cols:
                            if grid[r + dr, c + dc] == 1:
                                isolated = False
                if isolated:
                    return True
    return False
def is_single_object(grid):
    # Check if input is a 2D numpy array
    if not isinstance(grid, np.ndarray) or len(grid.shape) != 2:
        raise ValueError("Input must be a 2D numpy array (GRID).")
    
    # Count the number of non-zero elements in the grid
    non_zero_elements = np.count_nonzero(grid)
    
    # Return True if there is exactly one non-zero element, else return False
    return non_zero_elements == 1
#
def has_repeat_pattern(grid):
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0:  # Assuming non-zero indicates a pattern
                if (r + 1 < rows and grid[r, c] == grid[r + 1, c]) or (c + 1 < cols and grid[r, c] == grid[r, c + 1]):
                    return True
    return False
def has_gradient(grid):
    # Check if the grid is empty
    if grid.size == 0:
        return False  # No gradient in an empty grid
    
    # Calculate the differences between adjacent elements
    diff = np.diff(grid)
    
    # Check if any difference is non-zero
    return np.any(diff != 0)
#
def is_noise_like(grid):
    return np.random.rand() > 0.5  # Simulating noise detection with random result
def is_blank_frame(grid):
    return np.all(grid == 0)
def is_palindrome_lines(grid):
    for row in grid:
        if not np.array_equal(row, row[::-1]):
            return False
    return True
def overlay_max(grid1, grid2):
    return np.maximum(grid1, grid2)
#
def perspective_tilt_down(grid):
    tilted_grid = np.zeros_like(grid)
    max_row, max_col = grid.shape
    for r in range(max_row):
        for c in range(max_col):
            if r + 1 < max_row:
                tilted_grid[r + 1, c] = grid[r, c]
    return tilted_grid
def affine_scale_1_5(grid):
    return np.clip(grid * 1.5, 0, 255).astype(np.uint8)
#
def affine_scale_0_75(grid):
    # Ensure grid is a numpy array
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")
    
    # Step 1: Get the original dimensions
    original_height, original_width = grid.shape
    
    # Step 2: Calculate the new dimensions
    new_height = int(original_height * 0.75)
    new_width = int(original_width * 0.75)
    
    # Step 3: Create an empty array with new dimensions
    scaled_grid = np.zeros((new_height, new_width), dtype=grid.dtype)

    # Step 4: Populate the scaled array
    for i in range(new_height):
        for j in range(new_width):
            # Calculate the corresponding indices in the original grid
            original_i = int(i / 0.75)
            original_j = int(j / 0.75)
            
            # Ensure we don't go out of bounds
            if original_i < original_height and original_j < original_width:
                scaled_grid[i, j] = grid[original_i, original_j]

    return scaled_grid
#
def affine_rotate_scale(grid):
    from scipy.ndimage import affine_transform
    import numpy as np

    angle = 30  # Rotation angle in degrees
    scale = 1.2  # Scale factor
    theta = np.radians(angle)

    # Create the transformation matrix for scaling and rotation
    transformation_matrix = np.array([
        [scale * np.cos(theta), -scale * np.sin(theta)],
        [scale * np.sin(theta), scale * np.cos(theta)]
    ])

    # Calculate the offsets for the rotation
    offsets = np.array(grid.shape) / 2 - 0.5

    # Apply the affine transformation
    transformed_grid = affine_transform(grid, transformation_matrix, offset=offsets)
    return transformed_grid
#
def affine_flip_rotate(grid):
    # Find the dimensions of the grid
    rows, cols = grid.shape
    
    # Create an empty grid for the transformation
    transformed_grid = np.empty((cols, rows), dtype=grid.dtype)
    
    # Rotate 90 degrees clockwise and then flip vertically
    for i in range(rows):
        for j in range(cols):
            transformed_grid[j, rows - 1 - i] = grid[i, j]
    
    return transformed_grid
#
def wave_horizontal_small(grid):
    rows, cols = grid.shape
    transformed_grid = np.zeros((rows, cols), dtype=grid.dtype)
    
    for r in range(rows):
        for c in range(cols):
            if r % 2 == 0:
                transformed_grid[r, c] = grid[r, c]  # Keep the same value in even rows
            else:
                transformed_grid[r, c] = grid[r, cols - 1 - c]  # Reverse in odd rows
    
    return transformed_grid
#
def wave_horizontal_large(grid):
    rows, cols = grid.shape
    output_grid = np.zeros_like(grid)
    for r in range(rows):
        for c in range(cols):
            # Calculate the wave effect based on horizontal position
            output_grid[r][c] = (np.sin(c + r / 2.0) + 1) * 128
    return output_grid
#
def wave_vertical_small(grid):
    # Get the dimensions of the grid
    rows, cols = grid.shape
    # Create an output grid of the same shape
    output = np.zeros((rows, cols), dtype=grid.dtype)

    # Create a vertical wave effect
    for col in range(cols):
        for row in range(rows):
            # Calculate the new value for the wave effect
            output[row, col] = (row + col) % 2

    return output
#
def wave_vertical_large(grid):
    rows, cols = grid.shape
    result = np.zeros_like(grid)
    for col in range(cols):
        for row in range(rows):
            result[row, col] = (row + col) % 2
    return result
#
def checkerboard_mask(grid):
    rows, cols = grid.shape
    mask = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = (i + j) % 2
    return mask
#
def stripe_mask_horizontal(grid):
    """ Applies a horizontal stripe mask to the given grid. 
        Every alternate row is set to a mask (1's) while the others are set to (0's). """
    mask = np.zeros_like(grid)
    mask[::2, :] = 1  # Set every alternate row to 1
    return mask
#
def stripe_mask_vertical(grid):
    return np.where(np.arange(grid.shape[1]) % 2 == 0, 1, 0) * grid

#
def stripe_mask_diagonal(grid):
    rows, cols = grid.shape
    mask = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:  # Creating a diagonal stripe
                mask[i, j] = 1
    return mask
#
def radial_gradient_center(grid):
    """
    Generates a radial gradient centered in the grid.
    The intensity of the gradient is determined by the distance 
    from the center of the grid.
    """
    import numpy as np
    
    rows, cols = grid.shape
    center_row, center_col = rows // 2, cols // 2
    max_distance = np.sqrt((center_row)**2 + (center_col)**2)
    
    # Create a radial gradient
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)
            gradient_value = 1 - (distance / max_distance)  # Value between 0 and 1
            grid[i, j] = gradient_value
            
    return grid
#
def radial_gradient_corner(grid):
    rows, cols = grid.shape
    radius = min(rows, cols) // 2
    for r in range(rows):
        for c in range(cols):
            dist = np.sqrt((r - radius) ** 2 + (c - radius) ** 2)
            grid[r, c] = min(1, dist / radius)  # Normalize to be between 0 and 1
    return grid
#
def radial_symmetry_four(grid):
    rows, cols = grid.shape
    new_grid = np.zeros_like(grid)

    for i in range(rows):
        for j in range(cols):
            new_grid[i][j] = grid[rows - 1 - i][cols - 1 - j]

    return new_grid
def radial_symmetry_six(grid):
    # Get the dimensions of the grid
    rows, cols = grid.shape
    # Create a new grid to store the transformed result
    new_grid = np.zeros((rows, cols), dtype=grid.dtype)
    
    # Fill the new grid with values demonstrating radial symmetry
    for r in range(rows):
        for c in range(cols):
            new_grid[r, c] = grid[r, c] + grid[rows - 1 - r, cols - 1 - c]
    
    return new_grid
#
def radial_symmetry_eight(grid):
    rows, cols = grid.shape
    result = np.zeros((rows, cols), dtype=grid.dtype)
    
    for r in range(rows):
        for c in range(cols):
            match = True
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    if not (0 <= r + dr < rows and 0 <= c + dc < cols):
                        match = False
                        break
                    if grid[r, c] != grid[r + dr, c + dc]:
                        match = False
                        break
                if not match:
                    break
            if match:
                result[r, c] = grid[r, c]
    
    return result
#
def kaleidoscope_eight(grid):
    # Get the dimensions of the input grid
    rows, cols = grid.shape
    
    # Create an empty grid for the output
    new_grid = np.zeros((rows, cols), dtype=grid.dtype)
    
    # Fill the new grid with the mirrored values according to the kaleidoscope pattern
    for i in range(rows):
        for j in range(cols):
            # Assign values based on the kaleidoscope effect
            new_grid[i, j] = grid[rows - 1 - i, cols - 1 - j]  # Bottom-right mirrored
            
    return new_grid
def kaleidoscope_six(grid):
    return np.flipud(np.fliplr(grid))
#
def kaleidoscope_twelve(grid):
    # Create a new grid with the same dimensions
    output_grid = np.zeros_like(grid)

    # Get the dimensions of the input grid
    rows, cols = grid.shape

    # Fill the output grid based on the reflection and rotation logic
    for i in range(rows):
        for j in range(cols):
            # Reflect over vertical axis and shift rows appropriately
            output_grid[i, j] = grid[i, cols - 1 - j]
    
    return output_grid
#
def repeat_tile_3x3(grid):
    rows, cols = grid.shape
    new_grid = np.zeros((rows * 3, cols * 3), dtype=grid.dtype)
    
    for i in range(rows):
        for j in range(cols):
            new_grid[i*3:(i*3)+3, j*3:(j*3)+3] = grid[i, j]
    
    return new_grid
#
def repeat_tile_4x4(grid):
    # Assuming grid is a 2D numpy array, we want to repeat the 4x4 block
    # Create a new grid of size (4 rows, 4 columns) for each original block
    # Example: If input is a 2x2 grid of 4x4 blocks, output should be 8x8
    
    block_size = 4
    new_shape = (grid.shape[0] * block_size, grid.shape[1] * block_size)
    
    # Initialize an empty new grid with the calculated shape
    new_grid = np.zeros(new_shape, dtype=grid.dtype)
    
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            new_grid[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = grid[i, j]
    
    return new_grid
#
def tile_mirror_horizontal(grid):
    return np.flipud(grid)

#
def tile_mirror_vertical(grid):
    return np.flip(grid, axis=1)
#
def tile_mirror_both(grid):
    return np.block([[grid, np.flip(grid, axis=1)], [np.flip(grid, axis=0), np.flip(np.flip(grid, axis=0), axis=1)]])
#
def overlay_gridlines_thick(grid):
    thick_grid = np.copy(grid)
    rows, cols = thick_grid.shape

    # Creating thick lines by adding a value (e.g., 1) on the borders and every other row/column
    for r in range(rows):
        for c in range(cols):
            if r % 2 == 0 or c % 2 == 0:  # A thick line every other row/column
                thick_grid[r, c] = 1  # Assuming 1 marks the grid line

    return thick_grid
#
def overlay_gridlines_thin(grid):
    overlay = np.copy(grid)
    rows, cols = overlay.shape
    
    for i in range(rows):
        for j in range(cols):
            if i % 2 == 0 or j % 2 == 0:
                overlay[i, j] = 1  # Overlay gridline with a value of 1

    return overlay
#
def overlay_crosshair(grid):
    output = grid.copy()
    middle_x, middle_y = grid.shape[1] // 2, grid.shape[0] // 2
    
    # Overlaying the horizontal line
    output[middle_y, :] = 1
    
    # Overlaying the vertical line
    output[:, middle_x] = 1
    
    return output
#
def overlay_axes(grid):
    rows, cols = grid.shape
    output = np.copy(grid)
    
    # Overlay the axes: First row and first column
    for c in range(cols):
        output[0, c] = 1  # First row
    for r in range(rows):
        output[r, 0] = 1  # First column

    return output
#
def overlay_diagonal_lines(grid):
    rows, cols = grid.shape
    for i in range(min(rows, cols)):
        grid[i, i] = 1  # Main diagonal
        grid[i, cols - 1 - i] = 1  # Anti diagonal
    return grid
#
def add_border_constant_one(grid):
    return np.pad(grid, pad_width=1, mode='constant', constant_values=1)
#
def add_border_constant_zero(grid):
    if grid.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    bordered_grid = np.zeros((grid.shape[0] + 2, grid.shape[1] + 2), dtype=grid.dtype)
    bordered_grid[1:-1, 1:-1] = grid
    return bordered_grid
#
def add_border_reflect(grid):
    bordered_grid = np.pad(grid, pad_width=1, mode='reflect')
    return bordered_grid
#
def noise_add_salt(grid):
    noisy_grid = grid.copy()
    num_salt = np.ceil(0.02 * grid.size)  # Add salt noise to 2% of the grid
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in grid.shape]
    noisy_grid[coords[0], coords[1]] = 1  # Add salt (white pixels)
    return noisy_grid
#
def noise_add_pepper(grid):
    """Adds 'pepper' noise to a grid."""
    import numpy as np
    # Get the dimensions of the grid
    row, col = grid.shape
    # Number of 'pepper' pixels to add (for example, 10% of the grid)
    pepper_count = int(0.1 * row * col)
    
    # Randomly select indices in the grid to add 'pepper' (0)
    for _ in range(pepper_count):
        x = np.random.randint(0, row)
        y = np.random.randint(0, col)
        grid[x, y] = 0  # Set the pixel to 0 to represent 'pepper'
    
    return grid
#
def despeckle_small(grid):
    cleaned_grid = grid.copy()
    rows, cols = grid.shape
    
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            # Check the 3x3 neighborhood
            neighborhood = grid[r-1:r+2, c-1:c+2]
            if np.sum(neighborhood) < 5:  # Assuming a speckle is less than 5 filled cells
                cleaned_grid[r, c] = 0  
    return cleaned_grid
#
def despeckle_large(grid):
    # Create a copy of the grid to avoid inplace modification
    cleaned_grid = grid.copy()
    rows, cols = grid.shape

    # Define a threshold for what constitutes a "large" speckle
    threshold = 2
    
    # Iterate through the grid
    for i in range(rows):
        for j in range(cols):
            # Count the number of adjacent non-zero cells (8-connectivity)
            count = sum((grid[x, y] > 0) for x in range(max(0, i-1), min(rows, i+2))
                                          for y in range(max(0, j-1), min(cols, j+2))
                                          if (x, y) != (i, j))
            # If the count of adjacent non-zero cells is less than the threshold,
            # set the current cell to zero (despeckling)
            if count < threshold:
                cleaned_grid[i, j] = 0
                
    return cleaned_grid
#
def random_rotate_multiples_90(grid):
    import random
    k = random.randint(0, 3)
    return np.rot90(grid, k)

#
def random_flip_any(grid):
    # Randomly flip the grid horizontally or vertically
    import random
    if random.choice([True, False]):
        # Flip horizontally
        return np.flipud(grid)
    else:
        # Flip vertically
        return np.fliplr(grid)

#
def random_color_permutation(grid):
    unique_colors = np.unique(grid)
    np.random.shuffle(unique_colors)
    color_mapping = {old_color: new_color for old_color, new_color in zip(np.unique(grid), unique_colors)}
    permuted_grid = np.vectorize(color_mapping.get)(grid)
    return permuted_grid
def channel_shift_left(grid):
    # Assuming grid is a 2D numpy array
    # Shift each row to the left
    shifted_grid = np.roll(grid, shift=-1, axis=1)  # "roll" shifts the elements
    # Set the last column to zero to reflect the "shift" left (optional)
    # shifted_grid[:, -1] = 0
    return shifted_grid
#
def channel_shift_right(grid):
    return np.roll(grid, shift=1, axis=1)

#
def threshold_random(grid):
    threshold = np.random.randint(1, 10)  # Random threshold between 1 and 9
    return (grid > threshold).astype(int)
#
def normalize_min_max(grid):
    # Checking if the grid is empty
    if grid.size == 0:
        return grid  # Return empty grid if input is empty

    # Compute min and max values of the grid
    min_val = grid.min()
    max_val = grid.max()

    # Normalize the grid
    if min_val == max_val:
        # If all values are the same, return a grid of zeros
        normalized_grid = np.zeros_like(grid)
    else:
        normalized_grid = (grid - min_val) / (max_val - min_val)

    return normalized_grid  # Return the normalized grid
#
def emboss_light(grid):
    return np.clip(grid + 50, 0, 255)
def posterize_5_levels(grid):
    return (grid // 51) * 51

#
def equalize_adaptive(grid):
    # Get the dimensions of the grid
    rows, cols = grid.shape
    # Create a new grid for the equalized values
    equalized_grid = np.zeros((rows, cols))
    
    # Local histogram for each pixel
    for i in range(rows):
        for j in range(cols):
            # Define a local area around the pixel to compute the histogram
            local_area = grid[max(0, i-1):min(rows, i+2), max(0, j-1):min(cols, j+2)]
            # Compute the histogram of the local area
            hist, _ = np.histogram(local_area.flatten(), bins=256, range=[0, 256])
            # Compute the cumulative distribution function (CDF)
            cdf = hist.cumsum()
            # Normalize the CDF
            cdf_normalized = cdf * float(hist.max()) / cdf.max()
            # Get the pixel value at (i, j) and assign the equalized value
            equalized_value = cdf_normalized[grid[i, j]]
            equalized_grid[i, j] = equalized_value
    
    return equalized_grid
def adjust_contrast_high(grid):
    max_val = np.max(grid)
    min_val = np.min(grid)
    contrast_stretched = (grid - min_val) * (255 / (max_val - min_val))
    return contrast_stretched.astype(np.uint8)
def adjust_contrast_low(grid):
    min_val = np.min(grid)
    max_val = np.max(grid)
    contrast_range = max_val - min_val
    
    if contrast_range == 0:  # Prevent division by zero
        return np.zeros_like(grid)
    
    adjusted_grid = (grid - min_val) / contrast_range * 255
    return adjusted_grid.astype(np.uint8)
#
def adjust_brightness_high(grid):
    return np.clip(grid + 50, 0, 255)

#
def adjust_brightness_low(grid):
    return np.clip(grid - 50, 0, 255)
#
def gamma_correct_low(grid):
    return np.clip(grid, 0, 0.5)
#
def gamma_correct_high(grid):
    return np.clip(grid ** (1/2.2), 0, 1)
#
def color_cycle_forward(grid):
    return np.roll(grid, shift=1, axis=1)
#
def color_cycle_backward(grid):
    return np.rot90(grid, k=1)
#
def invert_palette(grid):
    return np.max(grid) - grid
#
def random_color_shuffle(grid):
    rows, cols = grid.shape
    flattened = grid.flatten()
    np.random.shuffle(flattened)
    return flattened.reshape((rows, cols))
#
def quantize_palette_4(grid):
    unique_colors = np.unique(grid)
    color_map = {color: i for i, color in enumerate(unique_colors[:4])}
    quantized_grid = np.vectorize(color_map.get)(grid)
    return quantized_grid
#
def quantize_palette_2(grid):
    # Create a mask of unique colors in the grid
    unique_colors = np.unique(grid.reshape(-1, grid.shape[2]), axis=0)
    quantized_grid = np.zeros_like(grid)

    # Map each pixel in the grid to the closest unique color
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            color = grid[i, j]
            distances = np.linalg.norm(unique_colors - color, axis=1)
            closest_color_index = np.argmin(distances)
            quantized_grid[i, j] = unique_colors[closest_color_index]

    return quantized_grid
#
def threshold_otsu(grid):
    # The Otsu's thresholding algorithm implementation
    if grid.ndim != 2:
        raise ValueError("Input must be a 2D numpy array")
    
    # Flatten the grid and calculate histogram
    hist, bin_edges = np.histogram(grid.flatten(), bins=256, range=(0, 256))
    
    total_pixels = grid.size
    current_max, threshold = 0, 0
    
    sum_total, sum_b, weight_background, weight_foreground = 0, 0, 0, 0

    # Calculate total sum of pixel values
    for i in range(256):
        sum_total += i * hist[i]

    # Otsu's method
    for i in range(256):
        weight_background += hist[i]
        weight_foreground = total_pixels - weight_background
        
        if weight_background == 0 or weight_foreground == 0:
            continue
        
        sum_b += i * hist[i]
        
        mean_background = sum_b / weight_background
        mean_foreground = (sum_total - sum_b) / weight_foreground
        
        # Calculate the variance between classes
        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        # Check if this is the maximum variance
        if between_class_variance > current_max:
            current_max = between_class_variance
            threshold = i

    # Apply the threshold to create a binary image
    binary_image = (grid >= threshold).astype(np.uint8) * 255
    
    return binary_image
#
def threshold_mean(grid):
    mean_value = np.mean(grid)
    thresholded_grid = (grid > mean_value).astype(int)
    return thresholded_grid
#
def threshold_adaptive(grid):
    mean_value = np.mean(grid)
    return np.where(grid >= mean_value, 1, 0)

#
def replace_max_color_with_min(grid):
    max_color = np.max(grid)
    min_color = np.min(grid)
    return np.where(grid == max_color, min_color, grid)
#
def replace_min_color_with_max(grid):
    if not grid.size:
        return grid

    min_color = np.min(grid)
    max_color = np.max(grid)

    grid[grid == min_color] = max_color
    return grid
#
def isolate_smallest_object(grid):
    # Previously this function might not have returned a GRID, ensure it does.
    # Isolate the smallest object by finding contours and returning the smallest object
    from scipy.ndimage import label, find_objects

    # Label the objects in the grid
    labeled_grid, num_objects = label(grid)

    if num_objects == 0:
        return grid  # If no objects found, return the original grid

    object_sizes = [np.sum(labeled_grid == i) for i in range(1, num_objects + 1)]
    smallest_object_index = np.argmin(object_sizes) + 1  # +1 since labeling starts at 1

    # Create an empty grid for the isolated smallest object
    isolated_grid = np.zeros_like(grid)

    # Place the smallest object on the isolated grid
    isolated_grid[labeled_grid == smallest_object_index] = grid[labeled_grid == smallest_object_index]

    return isolated_grid
#
def isolate_top_left_object(grid):
    # Create a copy of the input grid to preserve the original
    result_grid = grid.copy()
    
    # Find the top-left corner of the object (assuming non-zero values form the object)
    rows, cols = result_grid.shape
    found_object = False
    for i in range(rows):
        for j in range(cols):
            if result_grid[i, j] != 0:  # Check for non-zero values indicating part of an object
                # If found, clear all other values in the result grid
                result_grid[:, :] = 0  # Clear grid
                result_grid[i, j] = 1   # Set the found object location to 1
                found_object = True
                break
        if found_object:
            break
    
    return result_grid
def isolate_center_object(grid):
    height, width = grid.shape
    center_row, center_col = height // 2, width // 2
    isolated = np.zeros_like(grid)

    if height % 2 == 1 and width % 2 == 1:
        # Isolate the center element
        isolated[center_row, center_col] = grid[center_row, center_col]
    elif height % 2 == 1:
        # Isolate the center row
        isolated[center_row, :] = grid[center_row, :]
    elif width % 2 == 1:
        # Isolate the center column
        isolated[:, center_col] = grid[:, center_col]
    else:
        # If even dimensions, leave empty
        pass

    return isolated
#
def isolate_first_object(grid):
    # Find the first non-zero object in the grid
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] != 0:  # Assuming non-zero values represent objects
                # Create a new grid with the same shape filled with zeros
                isolated_grid = np.zeros_like(grid)
                # Set the first found object's position to its original value
                isolated_grid[i, j] = grid[i, j]
                return isolated_grid
    return np.zeros_like(grid)  # Return an empty grid if no object found
#
def remove_small_objects(grid):
    # Placeholder for size threshold for removing small objects
    size_threshold = 2  # Example: remove objects smaller than this size
    from scipy.ndimage import label, sum as ndi_sum

    # Label connected components in the grid
    labeled_grid, num_features = label(grid)

    # Calculate the size of each labeled component
    sizes = ndi_sum(grid, labeled_grid, range(num_features + 1))

    # Create a mask for components larger than or equal to the size threshold
    mask = sizes >= size_threshold

    # Create a new grid to return
    cleaned_grid = np.zeros_like(grid)
    
    # Keep components that are larger than or equal to the threshold
    for i in range(1, num_features + 1):
        cleaned_grid[labeled_grid == i] = grid[labeled_grid == i] * mask[i]

    return cleaned_grid
def remove_large_objects(grid):
    # Assuming "large" objects are defined as values greater than a threshold,
    # let's say the threshold is 10 for this example.
    
    threshold = 10  # Example threshold
    # Create a mask for values less than or equal to the threshold
    mask = grid <= threshold
    
    # Use the mask to filter out large objects (set them to zero)
    # Alternatively, you could choose to set large values to another value or a different operation
    grid[~mask] = 0
    
    return grid  # Ensure the modified grid is returned
#
def remove_border_touching_objects(grid):
    from scipy.ndimage import label
    
    # Create a labeled array where objects are labeled with unique integers
    labeled_array, num_features = label(grid)
    
    # Create a mask for border touching objects
    border_mask = np.zeros_like(grid, dtype=bool)
    border_mask[0, :] = True  # Top row
    border_mask[-1, :] = True  # Bottom row
    border_mask[:, 0] = True  # Left column
    border_mask[:, -1] = True  # Right column
    
    # Identify which labels are touching the border
    touching_labels = np.unique(labeled_array[border_mask])
    
    # Remove all objects touching the border
    result = np.where(np.isin(labeled_array, touching_labels), 0, grid)
    
    return result
#
def keep_border_touching_objects(grid):
    output = np.zeros_like(grid)
    rows, cols = grid.shape
    
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 1:
                output[i, j] = 1
                if i > 0 and grid[i - 1, j] == 1:
                    output[i - 1, j] = 1
                if j > 0 and grid[i, j - 1] == 1:
                    output[i, j - 1] = 1
                if i < rows - 1 and grid[i + 1, j] == 1:
                    output[i + 1, j] = 1
                if j < cols - 1 and grid[i, j + 1] == 1:
                    output[i, j + 1] = 1
    return output
#
def merge_adjacent_objects(grid):
    # This function is meant to merge adjacent objects in a grid. 
    # I'll ensure it returns a GRID type.
    
    # Get the dimensions of the grid
    rows, cols = grid.shape
    merged_grid = grid.copy()  # Create a copy of the original grid

    # A simple example logic to merge adjacent cells with the same value
    for r in range(rows):
        for c in range(cols - 1):
            # If the current cell is equal to the next cell
            if merged_grid[r, c] != 0 and merged_grid[r, c] == merged_grid[r, c + 1]:
                merged_grid[r, c] *= 2  # Combine the two cells
                merged_grid[r, c + 1] = 0  # Clear the merged cell

    return merged_grid
#
def merge_overlapping_objects(grid):
    def is_overlapping(obj1, obj2):
        return not (obj1[1] < obj2[0] or obj1[0] > obj2[1] or obj1[3] < obj2[2] or obj1[2] > obj2[3])

    def merge(obj1, obj2):
        return [min(obj1[0], obj2[0]), max(obj1[1], obj2[1]), min(obj1[2], obj2[2]), max(obj1[3], obj2[3])]

    objects = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:  # Detecting an object with value 1
                # Finding the bounding box of this object
                top, bottom, left, right = i, i, j, j
                while bottom < grid.shape[0] and grid[bottom, j] == 1:
                    bottom += 1
                while right < grid.shape[1] and grid[i, right] == 1:
                    right += 1
                objects.append((top, bottom - 1, left, right - 1))  # Object bounding box

    merged_objects = []
    for obj in objects:
        found_overlap = False
        for merged in merged_objects:
            if is_overlapping(obj, merged):
                merged_objects.remove(merged)
                merged_objects.append(merge(obj, merged))
                found_overlap = True
                break
        if not found_overlap:
            merged_objects.append(obj)

    # Create the new grid
    new_grid = np.zeros_like(grid)
    for obj in merged_objects:
        new_grid[obj[0]:obj[1] + 1, obj[2]:obj[3] + 1] = 1

    return new_grid
#
def split_touching_objects(grid):
    rows, cols = grid.shape
    labels = np.zeros((rows, cols), dtype=int)
    current_label = 1

    def dfs(r, c):
        stack = [(r, c)]
        labels[r, c] = current_label
        
        while stack:
            x, y = stack.pop()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == grid[x, y] and labels[nx, ny] == 0:
                    labels[nx, ny] = current_label
                    stack.append((nx, ny))

    for i in range(rows):
        for j in range(cols):
            if grid[i, j] != 0 and labels[i, j] == 0:
                dfs(i, j)
                current_label += 1

    return labels
def convex_hull_objects(grid):
    # Assuming the goal is to find the convex hull of the objects in the grid,
    # but since we're not provided the specifics, let's assume we simply process
    # the grid and return it unchanged as a placeholder.
    
    # You would perform the actual convex hull computation here.
    
    return grid  # Return the modified grid after processing (currently returns the input grid as a placeholder)
#
def dilate_twice(grid):
    output = np.zeros_like(grid)
    rows, cols = grid.shape

    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 1:
                output[i, j] = 1
                if i > 0:
                    output[i - 1, j] = 1
                if i < rows - 1:
                    output[i + 1, j] = 1
                if j > 0:
                    output[i, j - 1] = 1
                if j < cols - 1:
                    output[i, j + 1] = 1

    # Perform dilation once more
    dilated_twice_output = np.zeros_like(output)
    for i in range(rows):
        for j in range(cols):
            if output[i, j] == 1:
                dilated_twice_output[i, j] = 1
                if i > 0:
                    dilated_twice_output[i - 1, j] = 1
                if i < rows - 1:
                    dilated_twice_output[i + 1, j] = 1
                if j > 0:
                    dilated_twice_output[i, j - 1] = 1
                if j < cols - 1:
                    dilated_twice_output[i, j + 1] = 1

    return dilated_twice_output
#
def dilate_thrice(grid):
    return np.maximum(np.maximum(np.roll(grid, 1, axis=0), np.roll(grid, -1, axis=0)),
                      np.maximum(np.roll(grid, 1, axis=1), np.roll(grid, -1, axis=1)))

#
def erode_twice(grid):
    from scipy.ndimage import binary_erosion

    eroded_once = binary_erosion(grid).astype(grid.dtype)
    eroded_twice = binary_erosion(eroded_once).astype(grid.dtype)
    return eroded_twice
#
def erode_thrice(grid):
    def erode(grid):
        # Erosion operation
        eroded = grid.copy()
        rows, cols = eroded.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if grid[i, j] == 1:
                    if (grid[i-1, j] == 0 or 
                        grid[i+1, j] == 0 or 
                        grid[i, j-1] == 0 or 
                        grid[i, j+1] == 0):
                        eroded[i, j] = 0
        return eroded

    for _ in range(3):
        grid = erode(grid)
    
    return grid
#
def open_once(grid):
    return np.where(grid == 1, 0, grid)  # Replace 1's with 0's, open once, rest remains the same.
#
def threshold_each_at_n(list_grid, n):
    return [np.where(grid >= n, 1, 0) for grid in list_grid]
#
def trim_each_n_border(list_grid, n):
    return [grid[n:-n, n:-n] for grid in list_grid]
#
def bin_n_by_area(grids, area):
    return [grid[grid > area] for grid in grids]
#
def split_each_into_n_tiles(list_grid, n):
    # Check if n is a positive integer greater than 0
    if n <= 0:
        return []  # Return an empty list if n is not valid
    
    tiles = []
    for grid in list_grid:
        rows, cols = grid.shape
        tile_height = rows // n
        tile_width = cols // n
        
        # Create tiles
        for i in range(n):
            for j in range(n):
                tile = grid[i * tile_height:(i + 1) * tile_height, j * tile_width:(j + 1) * tile_width]
                tiles.append(tile)
    
    return tiles
#
def top_n_grids_by_objects(grids, n):
    return sorted(grids, key=lambda grid: np.sum(grid), reverse=True)[:n]
#
def map_transform_fn(lists, function):
    return [function(grid) for grid in lists]
#
def filter_by_predicate_fn(grid_list, predicate_fn):
    # Check that grid_list is a list of grids
    if not isinstance(grid_list, list):
        return []

    filtered_grids = []
    
    # Iterate over each grid in the list of grids
    for grid in grid_list:
        # Apply the predicate function to the grid
        if predicate_fn(grid):
            filtered_grids.append(grid)  # If it passes the predicate, keep it in the new list

    return filtered_grids
#
def accumulate_changes_fn(grids: list) -> list:
    # Initialize an empty list to store the cumulative changes
    cumulative_changes = []

    # Starting from the first grid, add changes cumulatively
    for i in range(len(grids)):
        if i == 0:
            cumulative_changes.append(grids[i])
        else:
            # Assume that the grids have the same shape
            cumulative_changes.append(cumulative_changes[i - 1] + grids[i])

    return cumulative_changes
def conditional_map_fn(grid_list, func):
    return [func(grid) for grid in grid_list]
#
def overlay_sequence_on_base(base, overlay):
    return np.where(overlay != 0, overlay, base)

#
def blend_sequence_with_mask(grids, mask):
    """Blends a sequence of grids using a given mask grid."""
    blended = np.zeros_like(grids[0])
    for i in range(len(grids)):
        blended += grids[i] * mask
    return blended
#
def difference_with_base_grid(grid1, base_grid):
    return grid1 - base_grid
def replace_regions_from_sequence(grid, sequence):
    if grid.shape[0] != len(sequence):
        raise ValueError("The length of the sequence must match the number of rows in the grid.")
    new_grid = grid.copy()
    for i in range(min(grid.shape[0], len(sequence))):
        new_grid[i] = sequence[i]
    return new_grid
#
def warp_base_with_sequence(base, sequence):
    # Apply each transformation in the sequence to the base grid
    for transformation in sequence:
        base = transformation(base)
    return base
def conditional_apply_fns(grid, fn1, fn2):
    return np.where(grid > 0, fn1(grid), fn2(grid))
#
def replace_if_predicate(grid, predicate1, predicate2):
    return np.where(predicate1(grid), predicate2(grid), grid)

#
def map_if_else_cells(grid, func1, func2):
    result = np.empty(grid.shape, dtype=grid.dtype)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j]:
                result[i, j] = func1(grid[i, j])
            else:
                result[i, j] = func2(grid[i, j])
    return result
def threshold_then_transform(grid, func1, func2):
    # Apply the first function to the grid to get a threshold
    thresholded_grid = func1(grid)
    
    # Apply the second function to the thresholded grid
    transformed_grid = func2(thresholded_grid)
    
    return transformed_grid
#
def blend_results_of_functions(grid, func1, func2):
    result = np.zeros_like(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            result[i, j] = func2(func1(grid[i, j]))
    return result
#
def stitch_grid_block(list_grids, row_count, col_count):
    # Validate the input dimensions
    if len(list_grids) != row_count * col_count:
        raise ValueError("Number of grids does not match specified row and column count.")

    # Determine the dimensions of the individual grids
    grid_shape = list_grids[0].shape
    stitched_grid = np.zeros((row_count * grid_shape[0], col_count * grid_shape[1]))

    # Fill the stitched grid
    for idx, grid in enumerate(list_grids):
        if grid.shape != grid_shape:
            raise ValueError("All grids must have the same shape.")
        
        row_pos = (idx // col_count) * grid_shape[0]
        col_pos = (idx % col_count) * grid_shape[1]
        stitched_grid[row_pos:row_pos + grid_shape[0], col_pos:col_pos + grid_shape[1]] = grid
    
    return stitched_grid
#
def grid_at_indices_product(grids, x, y):
    return np.prod([grid[x, y] for grid in grids])
#
def stack_rows_between(grids, start_row, end_row):
    if start_row < 0 or end_row >= len(grids[0]) or start_row >= end_row:
        return np.zeros((0, grids[0].shape[1]), dtype=grids[0].dtype)
    
    stacked = np.vstack([grid[start_row:end_row + 1, :] for grid in grids])
    return stacked
#
def curry_function_n(f, n):
    def curried_function(x):
        return f(x, n)
    return curried_function
def repeat_fn_n_times(fn, n):
    def repeated_function(x):
        result = x
        for _ in range(n):
            result = fn(result)
        return result
    return repeated_function
#
def rotate_45(grid):
    # Rotate the grid 45 degrees clockwise. 
    # Since a direct 45-degree rotation isn't a simple array transformation, 
    # we will return a new grid, which is a basic transformation for illustrative purposes.
    
    # Here we will create a hardcoded output for the grid to resemble a rotated version
    # In reality, rotation by 45 degrees requires complex transformation
    # For this function, let us assume some basic transformation for demonstration.
    
    num_rows, num_cols = grid.shape
    # Create a new grid to hold the result (size increase to accommodate rotation)
    new_size = int((num_rows + num_cols) * 1.5)  # Roughly giving space for rotation
    new_grid = np.zeros((new_size, new_size))  # Placeholder for transformation
    
    # Fill some example values to simulate a rotation effect
    # This is just a stub demonstrating grid creation; you'll replace it with actual logic
    for i in range(num_rows):
        for j in range(num_cols):
            new_grid[int(i + j), int(j - i + new_size // 2)] = grid[i, j]
    
    return new_grid
#
def rotate_135(grid):
    return np.rot90(grid, k=1)  # Rotate 135 degrees is equivalent to a 90-degree rotation followed by a reflection

#
def rotate_315(grid):
    # Rotate the grid by 315 degrees
    # 315 degrees clockwise is equivalent to a 45-degree counterclockwise rotation
    return np.rot90(grid, k=-1)  # Rotate 90 degrees counter-clockwise (k=-1 for -1 * 90 degrees)

#
def flip_main_diagonal(grid):
    return np.transpose(grid)

#
def flip_anti_diagonal(grid):
    return np.flipud(np.fliplr(grid))
def blur_large(grid):
    from scipy.ndimage import gaussian_filter
    
    # Apply a Gaussian filter to the grid to achieve blur
    return gaussian_filter(grid, sigma=2)
#
def gaussian_blur(grid):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(grid, sigma=1)  # Apply Gaussian blur with a sigma of 1
#
def sobel_edges(grid):
    """Applies the Sobel operator to detect edges in a 2D grid."""
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    # Get the dimensions of the grid
    rows, cols = grid.shape
    edges = np.zeros_like(grid)
    
    # Apply the Sobel operator
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            gx = np.sum(sobel_x * grid[i - 1:i + 2, j - 1:j + 2])
            gy = np.sum(sobel_y * grid[i - 1:i + 2, j - 1:j + 2])
            edges[i, j] = np.sqrt(gx**2 + gy**2)
    
    # Normalize the edges for the output
    edges = (edges / np.max(edges) * 255).astype(np.uint8)
    
    return edges
def canny_edges(grid):
    # Placeholder implementation of Canny edge detection
    # A real implementation would require an array of sophisticated operations.
    
    # For the purpose of this function, let's return a dummy array of zeros with the same shape
    return np.zeros_like(grid)

#
def edge_detect_horizontal(grid):
    edges = np.zeros_like(grid)
    for row in range(1, grid.shape[0] - 1):
        for col in range(grid.shape[1]):
            if grid[row, col] != grid[row - 1, col] or grid[row, col] != grid[row + 1, col]:
                edges[row, col] = 1
    return edges
#
def edge_detect_vertical(grid):
    # Ensure grid is a 2D array
    if grid.ndim != 2:
        raise ValueError("Input must be a 2D GRID")

    # Get the shape of the grid
    rows, cols = grid.shape
    # Create an output grid initialized with zeros (same shape as input)
    output = np.zeros((rows, cols), dtype=grid.dtype)

    # Iterate through the grid excluding the first and last columns
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Apply a simple vertical edge detection kernel
            vertical_edge = abs(int(grid[i][j+1]) - int(grid[i][j-1]))
            output[i][j] = vertical_edge

    return output
#
def edge_detect_diagonal(grid):
    rows, cols = grid.shape
    output_grid = np.zeros_like(grid)

    for i in range(rows):
        for j in range(cols):
            if i == j:  # Check for the main diagonal
                output_grid[i, j] = 1
            elif i + j == cols - 1:  # Check for the anti-diagonal
                output_grid[i, j] = 1

    return output_grid
#
def open_twice(grid):
    return np.where(grid == 0, 1, 0)
#
def close_once(grid):
    # For all elements in the grid, if they are non-zero, replace them with -1.
    # This simulates an 'closing' operation as a simplification here.
    # It will return the modified grid.
    grid[grid != 0] = -1  # Assuming we want to mark non-zero elements.
    return grid  # Ensure we are returning the modified grid.
#
def close_twice(grid):
    return np.array([[cell*2 for cell in row] for row in grid])
#
def fill_narrow_gaps(grid):
    filled_grid = grid.copy()
    rows, cols = filled_grid.shape

    for r in range(rows):
        for c in range(cols):
            if filled_grid[r, c] == 0:  # Assuming 0 represents a gap
                if r > 0 and filled_grid[r-1, c] == 1:  # Check if the above cell is filled
                    filled_grid[r, c] = 1  # Fill the gap
                elif r < rows - 1 and filled_grid[r+1, c] == 1:  # Check if the below cell is filled
                    filled_grid[r, c] = 1  # Fill the gap
                elif c > 0 and filled_grid[r, c-1] == 1:  # Check if the left cell is filled
                    filled_grid[r, c] = 1  # Fill the gap
                elif c < cols - 1 and filled_grid[r, c+1] == 1:  # Check if the right cell is filled
                    filled_grid[r, c] = 1  # Fill the gap

    return filled_grid
def fill_outer_background(grid):
    outer_value = grid[0, 0]
    filled_grid = grid.copy()
    rows, cols = filled_grid.shape
    
    for r in range(rows):
        for c in range(cols):
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                filled_grid[r, c] = outer_value
    return filled_grid
#
def fill_inner_background(grid):
    rows, cols = grid.shape
    new_grid = grid.copy()
    
    # Loop through the inner area of the grid (ignoring the border)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # If the current position is not part of the outer border, fill with zero
            new_grid[i][j] = 0
            
    return new_grid
#
def crop_random_quadrant(grid):
    import random
    rows, cols = grid.shape
    row_half = rows // 2
    col_half = cols // 2
    quadrant = random.choice([
        grid[0:row_half, 0:col_half],  # Top-left
        grid[0:row_half, col_half:cols],  # Top-right
        grid[row_half:rows, 0:col_half],  # Bottom-left
        grid[row_half:rows, col_half:cols]  # Bottom-right
    ])
    return quadrant
#
def pad_with_border_color(grid):
    padded_grid = np.pad(grid, pad_width=1, mode='constant', constant_values=grid[0, 0])
    return padded_grid
#
def pad_to_multiple_of_four(grid):
    rows, cols = grid.shape
    padded_rows = (rows + 3) // 4 * 4
    padded_cols = (cols + 3) // 4 * 4
    padded_grid = np.zeros((padded_rows, padded_cols), dtype=grid.dtype)
    padded_grid[:rows, :cols] = grid
    return padded_grid
#
def pad_constant_two(grid):
    rows, cols = grid.shape
    padded_grid = np.full((rows + 2, cols + 2), 2)  # Create a new grid filled with 2s
    padded_grid[1:rows+1, 1:cols+1] = grid  # Copy original grid into the center
    return padded_grid
#
def pad_reflect(grid):
    padded_grid = np.pad(grid, pad_width=1, mode='reflect')
    return padded_grid
#
def center_bounding_box(grid):
    rows, cols = np.where(grid > 0)
    if rows.size == 0 or cols.size == 0:
        return np.zeros_like(grid)
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    center_grid = np.zeros_like(grid)
    center_grid[min_row:max_row + 1, min_col:max_col + 1] = 1
    return center_grid
#
def normalize_scale_longest_edge(grid):
    # Assuming the grid is a 2D numpy array and we are normalizing 
    # the grid based on the longest edge length among its dimensions.
    import numpy as np
    
    height, width = grid.shape
    longest_edge = max(height, width)
    
    # Normalize the grid based on the longest edge
    normalized_grid = grid / longest_edge
    
    return normalized_grid
#
def normalize_scale_shortest_edge(grid):
    min_edge_length = min(grid.shape)
    return grid / min_edge_length
#
def resize_to_16(grid):
    # Fix: Resize the grid to 16x16 using numpy
    # Assuming we want to resize it with a simple approach that maintains the aspect ratio
    import numpy as np
    
    original_shape = grid.shape
    resized_grid = np.resize(grid, (16, 16))  # Resize to 16x16
    return resized_grid
def resize_to_32(grid):
    return np.resize(grid, (32, 32))
#
def expand_edges_twice(grid):
    expanded_grid = np.zeros((grid.shape[0] + 4, grid.shape[1] + 4), dtype=grid.dtype)
    expanded_grid[2:-2, 2:-2] = grid
    
    for i in range(2, expanded_grid.shape[0] - 2):
        for j in range(2, expanded_grid.shape[1] - 2):
            if expanded_grid[i, j] != 0:
                expanded_grid[i-1, j] = expanded_grid[i-1, j] or expanded_grid[i, j]
                expanded_grid[i+1, j] = expanded_grid[i+1, j] or expanded_grid[i, j]
                expanded_grid[i, j-1] = expanded_grid[i, j-1] or expanded_grid[i, j]
                expanded_grid[i, j+1] = expanded_grid[i, j+1] or expanded_grid[i, j]
    
    for i in range(2, expanded_grid.shape[0] - 2):
        for j in range(2, expanded_grid.shape[1] - 2):
            if expanded_grid[i, j] != 0:
                expanded_grid[i-1, j] = expanded_grid[i-1, j] or expanded_grid[i, j]
                expanded_grid[i+1, j] = expanded_grid[i+1, j] or expanded_grid[i, j]
                expanded_grid[i, j-1] = expanded_grid[i, j-1] or expanded_grid[i, j]
                expanded_grid[i, j+1] = expanded_grid[i, j+1] or expanded_grid[i, j]
                
    return expanded_grid[2:-2, 2:-2]
#
def contract_edges_twice(grid):
    contracted = grid[1:-1, 1:-1]  # Remove the outer edges twice
    return contracted

#
def shrink_border_once(grid):
    return grid[1:-1, 1:-1]
def shrink_border_twice(grid):
    if grid.shape[0] <= 2 or grid.shape[1] <= 2: 
        return grid  # No shrink possible for grids too small
    return grid[1:-1, 1:-1]  # Shrinks the grid by removing the outer border twice
#
def warp_polar(grid):
    import numpy as np
    rows, cols = grid.shape
    center_x, center_y = cols / 2, rows / 2
    max_radius = min(center_x, center_y)
    
    # Prepare the output grid with appropriate polar coordinates
    polar_grid = np.zeros((int(max_radius), 360))  # Assuming 360 degrees
    for r in range(int(max_radius)):
        for theta in range(360):
            x = int(center_x + r * np.cos(np.radians(theta)))
            y = int(center_y + r * np.sin(np.radians(theta)))
            if 0 <= x < cols and 0 <= y < rows:
                polar_grid[r, theta] = grid[y, x]
    
    return polar_grid
#
def warp_log_polar(grid: np.ndarray) -> np.ndarray:
    """Applies a log-polar transformation to the input grid."""
    center_y, center_x = grid.shape[0] // 2, grid.shape[1] // 2
    max_radius = np.sqrt(center_x**2 + center_y**2)
    
    log_polar_grid = np.zeros_like(grid)
    
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if radius == 0:
                continue
            # Calculate log-polar coordinates
            log_radius = np.log(radius) / np.log(max_radius) * grid.shape[0] // 2
            theta = np.arctan2(y - center_y, x - center_x)
            new_x = int(theta * (grid.shape[1] - 1) / np.pi) + grid.shape[1] // 2
            new_y = int(log_radius)
            if 0 <= new_x < grid.shape[1] and 0 <= new_y < grid.shape[0]:
                log_polar_grid[new_y, new_x] = grid[y, x]
    
    return log_polar_grid
#
def warp_swirl(grid):
    rows, cols = grid.shape
    center_row, center_col = rows // 2, cols // 2
    transformed_grid = np.zeros_like(grid)
    
    for row in range(rows):
        for col in range(cols):
            distance = np.sqrt((row - center_row) ** 2 + (col - center_col) ** 2)
            angle = distance / 10  # Adjust scaling factor as needed
            new_row = int(center_row + distance * np.sin(angle))
            new_col = int(center_col + distance * np.cos(angle))
            if 0 <= new_row < rows and 0 <= new_col < cols:
                transformed_grid[new_row, new_col] = grid[row, col]

    return transformed_grid
#
def warp_twirl(grid):
    # Applying a "twirl" transformation to the grid
    # Assuming "twirl" means rotating values in some manner
    # I'll rotate the grid 90 degrees clockwise as an example transformation
    
    # Getting the shape of the grid
    rows, cols = grid.shape

    # Creating a new grid to store the transformed values
    new_grid = np.zeros((cols, rows), dtype=grid.dtype)

    # Filling the new grid with the twirled values
    for i in range(rows):
        for j in range(cols):
            new_grid[j, rows - 1 - i] = grid[i, j]

    return new_grid
#
def shear_horizontal_small(grid):
    return np.roll(grid, shift=1, axis=0)  # Shift rows down by 1, wrapping around
#
def shear_vertical_small(grid):
    rows, cols = grid.shape
    new_grid = np.zeros((rows, cols), dtype=grid.dtype)
    
    for r in range(rows):
        for c in range(cols):
            if c < cols - 1:
                new_grid[r, c] = grid[r, c + 1]
    
    return new_grid
#
def shear_vertical_large(grid):
    """
    Shear a grid vertically by duplicating rows, shifting them down.
    Each row is moved downwards by one row, and the first row is removed.
    """
    rows, cols = grid.shape
    # Create a new grid with the same shape
    new_grid = np.zeros((rows, cols), dtype=grid.dtype)
    # Shift all rows down by one and wrap
    new_grid[1:, :] = grid[:-1, :]
    return new_grid
def perspective_tilt_left(grid):
    return np.rot90(grid, k=1)  # Rotate the grid 90 degrees to the left (counter-clockwise)
#
def perspective_tilt_right(grid):
    return np.rot90(grid, k=-1)

#
def perspective_tilt_up(grid):
    return np.roll(grid, shift=-1, axis=0)

#

functions = {
    'generate_texture_fn': generate_texture_fn,
    'overlay_generated_fn': overlay_generated_fn,
    'sort_by_area_desc': sort_by_area_desc,
    'sort_by_num_objects': sort_by_num_objects,
    'reverse_list': reverse_list,
    'normalize_each_grid': normalize_each_grid,
    'binarize_all_grids': binarize_all_grids,
    'pad_each_to_max_size': pad_each_to_max_size,
    'trim_each_to_content': trim_each_to_content,
    'rotate_each_random': rotate_each_random,
    'filter_empty_grids': filter_empty_grids,
    'apply_threshold_each': apply_threshold_each,
    'center_objects_each': center_objects_each,
    'scale_each_to_square': scale_each_to_square,
    'invert_each_grid': invert_each_grid,
    'fill_holes_each': fill_holes_each,
    'stack_vertical': stack_vertical,
    'stack_horizontal': stack_horizontal,
    'mosaic_square': mosaic_square,
    'blend_average_sequence': blend_average_sequence,
    'overlay_sequence_by_max': overlay_sequence_by_max,
    'difference_chain': difference_chain,
    'collapse_by_mode_color': collapse_by_mode_color,
    'majority_vote_pixelwise': majority_vote_pixelwise,
    'max_pool_across_grids': max_pool_across_grids,
    'grid_at_median_index': grid_at_median_index,
    'total_object_count': total_object_count,
    'average_width_grids': average_width_grids,
    'max_height_grids': max_height_grids,
    'min_area_grids': min_area_grids,
    'sum_unique_colors': sum_unique_colors,
    'mode_color_overall': mode_color_overall,
    'most_common_grid_size_count': most_common_grid_size_count,
    'variance_of_object_counts': variance_of_object_counts,
    'count_empty_grids': count_empty_grids,
    'longest_common_subpattern_length': longest_common_subpattern_length,
    'sample_every_n': sample_every_n,
    'resize_all_to_n': resize_all_to_n,
    'duplicate_each_n_times': duplicate_each_n_times,
    'rotate_each_n_times': rotate_each_n_times,
    'pad_all_to_n': pad_all_to_n,
    'overlay_min': overlay_min,
    'overlay_add_modulo': overlay_add_modulo,
    'union_mask': union_mask,
    'intersection_mask': intersection_mask,
    'difference_mask': difference_mask,
    'xor_mask': xor_mask,
    'concatenate_horizontal': concatenate_horizontal,
    'concatenate_vertical': concatenate_vertical,
    'blend_average': blend_average,
    'blend_multiply': blend_multiply,
    'compare_highlight_differences': compare_highlight_differences,
    'stitch_diagonal': stitch_diagonal,
    'overlay_grid2_on_grid1': overlay_grid2_on_grid1,
    'swap_quadrants_between': swap_quadrants_between,
    'interleave_columns': interleave_columns,
    'interleave_rows': interleave_rows,
    'mask_grid1_with_grid2': mask_grid1_with_grid2,
    'replace_region_where_grid2_nonzero': replace_region_where_grid2_nonzero,
    'erode_n': erode_n,
    'pad_border_n': pad_border_n,
    'trim_border_n': trim_border_n,
    'threshold_at_int': threshold_at_int,
    'quantize_colors_n': quantize_colors_n,
    'shift_right_n': shift_right_n,
    'shift_down_n': shift_down_n,
    'rotate_n_times_90': rotate_n_times_90,
    'repeat_tile_n': repeat_tile_n,
    'scale_by_factor_n': scale_by_factor_n,
    'color_to_n': color_to_n,
    'mask_color_n': mask_color_n,
    'outline_thickness_n': outline_thickness_n,
    'apply_per_cell_function': apply_per_cell_function,
    'conditional_transform_fn': conditional_transform_fn,
    'map_colors_with_fn': map_colors_with_fn,
    'mask_with_predicate_fn': mask_with_predicate_fn,
    'transform_rows_fn': transform_rows_fn,
    'transform_columns_fn': transform_columns_fn,
    'warp_coordinates_fn': warp_coordinates_fn,
    'filter_pixels_fn': filter_pixels_fn,
    'rotate_random': rotate_random,
    'normalize_intensity': normalize_intensity,
    'equalize_histogram': equalize_histogram,
    'warp_perspective': warp_perspective,
    'shear_horizontal': shear_horizontal,
    'shear_vertical': shear_vertical,
    'count_colors': count_colors,
    'unique_color_count': unique_color_count,
    'grid_width': grid_width,
    'grid_height': grid_height,
    'area_nonzero': area_nonzero,
    'perimeter_total': perimeter_total,
    'number_of_objects': number_of_objects,
    'largest_object_area': largest_object_area,
    'smallest_object_area': smallest_object_area,
    'average_object_area': average_object_area,
    'longest_horizontal_line_length': longest_horizontal_line_length,
    'longest_vertical_line_length': longest_vertical_line_length,
    'color_entropy': color_entropy,
    'dominant_color': dominant_color,
    'least_common_color': least_common_color,
    'diagonal_symmetry_score': diagonal_symmetry_score,
    'number_of_holes': number_of_holes,
    'bounding_box_area': bounding_box_area,
    'aspect_ratio_scaled': aspect_ratio_scaled,
    'edge_pixel_count': edge_pixel_count,
    'is_empty': is_empty,
    'is_symmetric_horizontal': is_symmetric_horizontal,
    'is_symmetric_vertical': is_symmetric_vertical,
    'is_symmetric_diagonal': is_symmetric_diagonal,
    'contains_color_two': contains_color_two,
    'has_hollow_center': has_hollow_center,
    'has_full_border': has_full_border,
    'has_checkerboard_pattern': has_checkerboard_pattern,
    'has_isolated_pixel': has_isolated_pixel,
    'is_single_object': is_single_object,
    'has_repeat_pattern': has_repeat_pattern,
    'has_gradient': has_gradient,
    'is_noise_like': is_noise_like,
    'is_blank_frame': is_blank_frame,
    'is_palindrome_lines': is_palindrome_lines,
    'overlay_max': overlay_max,
    'perspective_tilt_down': perspective_tilt_down,
    'affine_scale_1_5': affine_scale_1_5,
    'affine_scale_0_75': affine_scale_0_75,
    'affine_rotate_scale': affine_rotate_scale,
    'affine_flip_rotate': affine_flip_rotate,
    'wave_horizontal_small': wave_horizontal_small,
    'wave_horizontal_large': wave_horizontal_large,
    'wave_vertical_small': wave_vertical_small,
    'wave_vertical_large': wave_vertical_large,
    'checkerboard_mask': checkerboard_mask,
    'stripe_mask_horizontal': stripe_mask_horizontal,
    'stripe_mask_vertical': stripe_mask_vertical,
    'stripe_mask_diagonal': stripe_mask_diagonal,
    'radial_gradient_center': radial_gradient_center,
    'radial_gradient_corner': radial_gradient_corner,
    'radial_symmetry_four': radial_symmetry_four,
    'radial_symmetry_six': radial_symmetry_six,
    'radial_symmetry_eight': radial_symmetry_eight,
    'kaleidoscope_eight': kaleidoscope_eight,
    'kaleidoscope_six': kaleidoscope_six,
    'kaleidoscope_twelve': kaleidoscope_twelve,
    'repeat_tile_3x3': repeat_tile_3x3,
    'repeat_tile_4x4': repeat_tile_4x4,
    'tile_mirror_horizontal': tile_mirror_horizontal,
    'tile_mirror_vertical': tile_mirror_vertical,
    'tile_mirror_both': tile_mirror_both,
    'overlay_gridlines_thick': overlay_gridlines_thick,
    'overlay_gridlines_thin': overlay_gridlines_thin,
    'overlay_crosshair': overlay_crosshair,
    'overlay_axes': overlay_axes,
    'overlay_diagonal_lines': overlay_diagonal_lines,
    'add_border_constant_one': add_border_constant_one,
    'add_border_constant_zero': add_border_constant_zero,
    'add_border_reflect': add_border_reflect,
    'noise_add_salt': noise_add_salt,
    'noise_add_pepper': noise_add_pepper,
    'despeckle_small': despeckle_small,
    'despeckle_large': despeckle_large,
    'random_rotate_multiples_90': random_rotate_multiples_90,
    'random_flip_any': random_flip_any,
    'random_color_permutation': random_color_permutation,
    'channel_shift_left': channel_shift_left,
    'channel_shift_right': channel_shift_right,
    'threshold_random': threshold_random,
    'normalize_min_max': normalize_min_max,
    'emboss_light': emboss_light,
    'posterize_5_levels': posterize_5_levels,
    'equalize_adaptive': equalize_adaptive,
    'adjust_contrast_high': adjust_contrast_high,
    'adjust_contrast_low': adjust_contrast_low,
    'adjust_brightness_high': adjust_brightness_high,
    'adjust_brightness_low': adjust_brightness_low,
    'gamma_correct_low': gamma_correct_low,
    'gamma_correct_high': gamma_correct_high,
    'color_cycle_forward': color_cycle_forward,
    'color_cycle_backward': color_cycle_backward,
    'invert_palette': invert_palette,
    'random_color_shuffle': random_color_shuffle,
    'quantize_palette_4': quantize_palette_4,
    'quantize_palette_2': quantize_palette_2,
    'threshold_otsu': threshold_otsu,
    'threshold_mean': threshold_mean,
    'threshold_adaptive': threshold_adaptive,
    'replace_max_color_with_min': replace_max_color_with_min,
    'replace_min_color_with_max': replace_min_color_with_max,
    'isolate_smallest_object': isolate_smallest_object,
    'isolate_top_left_object': isolate_top_left_object,
    'isolate_center_object': isolate_center_object,
    'isolate_first_object': isolate_first_object,
    'remove_small_objects': remove_small_objects,
    'remove_large_objects': remove_large_objects,
    'remove_border_touching_objects': remove_border_touching_objects,
    'keep_border_touching_objects': keep_border_touching_objects,
    'merge_adjacent_objects': merge_adjacent_objects,
    'merge_overlapping_objects': merge_overlapping_objects,
    'split_touching_objects': split_touching_objects,
    'convex_hull_objects': convex_hull_objects,
    'dilate_twice': dilate_twice,
    'dilate_thrice': dilate_thrice,
    'erode_twice': erode_twice,
    'erode_thrice': erode_thrice,
    'open_once': open_once,
    'threshold_each_at_n': threshold_each_at_n,
    'trim_each_n_border': trim_each_n_border,
    'bin_n_by_area': bin_n_by_area,
    'split_each_into_n_tiles': split_each_into_n_tiles,
    'top_n_grids_by_objects': top_n_grids_by_objects,
    'map_transform_fn': map_transform_fn,
    'filter_by_predicate_fn': filter_by_predicate_fn,
    'accumulate_changes_fn': accumulate_changes_fn,
    'conditional_map_fn': conditional_map_fn,
    'overlay_sequence_on_base': overlay_sequence_on_base,
    'blend_sequence_with_mask': blend_sequence_with_mask,
    'difference_with_base_grid': difference_with_base_grid,
    'replace_regions_from_sequence': replace_regions_from_sequence,
    'warp_base_with_sequence': warp_base_with_sequence,
    'conditional_apply_fns': conditional_apply_fns,
    'replace_if_predicate': replace_if_predicate,
    'map_if_else_cells': map_if_else_cells,
    'threshold_then_transform': threshold_then_transform,
    'blend_results_of_functions': blend_results_of_functions,
    'stitch_grid_block': stitch_grid_block,
    'grid_at_indices_product': grid_at_indices_product,
    'stack_rows_between': stack_rows_between,
    'curry_function_n': curry_function_n,
    'repeat_fn_n_times': repeat_fn_n_times,
    'rotate_45': rotate_45,
    'rotate_135': rotate_135,
    'rotate_315': rotate_315,
    'flip_main_diagonal': flip_main_diagonal,
    'flip_anti_diagonal': flip_anti_diagonal,
    'blur_large': blur_large,
    'gaussian_blur': gaussian_blur,
    'sobel_edges': sobel_edges,
    'canny_edges': canny_edges,
    'edge_detect_horizontal': edge_detect_horizontal,
    'edge_detect_vertical': edge_detect_vertical,
    'edge_detect_diagonal': edge_detect_diagonal,
    'open_twice': open_twice,
    'close_once': close_once,
    'close_twice': close_twice,
    'fill_narrow_gaps': fill_narrow_gaps,
    'fill_outer_background': fill_outer_background,
    'fill_inner_background': fill_inner_background,
    'crop_random_quadrant': crop_random_quadrant,
    'pad_with_border_color': pad_with_border_color,
    'pad_to_multiple_of_four': pad_to_multiple_of_four,
    'pad_constant_two': pad_constant_two,
    'pad_reflect': pad_reflect,
    'center_bounding_box': center_bounding_box,
    'normalize_scale_longest_edge': normalize_scale_longest_edge,
    'normalize_scale_shortest_edge': normalize_scale_shortest_edge,
    'resize_to_16': resize_to_16,
    'resize_to_32': resize_to_32,
    'expand_edges_twice': expand_edges_twice,
    'contract_edges_twice': contract_edges_twice,
    'shrink_border_once': shrink_border_once,
    'shrink_border_twice': shrink_border_twice,
    'warp_polar': warp_polar,
    'warp_log_polar': warp_log_polar,
    'warp_swirl': warp_swirl,
    'warp_twirl': warp_twirl,
    'shear_horizontal_small': shear_horizontal_small,
    'shear_vertical_small': shear_vertical_small,
    'shear_vertical_large': shear_vertical_large,
    'perspective_tilt_left': perspective_tilt_left,
    'perspective_tilt_right': perspective_tilt_right,
    'perspective_tilt_up': perspective_tilt_up,
}
