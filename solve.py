import numpy as np
from PIL import Image
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
from ortools.sat.python import cp_model
import time

""" Constants """

COLOR_THRESHOLD = 20
PEAK_THRESHOLD = 0.8
GRID_THRESHOLD = 0.5
MIN_PEAK_DISTANCE = 10
BORDER_WIDTH = 0.1

""" Visualization functions """

# Print a 2D array.
def print_array(array):
    print()
    for row in array:
        print(row)

# Visualize the board with colors and queens.
def visualize_board(grid_colors, board):
    global N
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title("LinkedIn Queens Solver")
    ax.add_patch(patches.Rectangle((-BORDER_WIDTH, -BORDER_WIDTH), N + 2 * BORDER_WIDTH, N + 2 * BORDER_WIDTH,
                                    facecolor='black', edgecolor='black', zorder=0))
    ax.set_xlim(-BORDER_WIDTH, N + BORDER_WIDTH)
    ax.set_ylim(-BORDER_WIDTH, N + BORDER_WIDTH)
    plt.gca().invert_yaxis()
    # Colored tiles
    for r in range(N):
        for c in range(N):
            color = grid_colors[r][c]
            rect = patches.Rectangle((c, r), 1, 1, facecolor=np.array(color) / 255, edgecolor='black')
            ax.add_patch(rect)
    # Queens visuals
    for r in range(N):
        for c in range(N):
            if board[r][c] == "Q":
                queen_color = 'black' if np.mean(grid_colors[r][c]) > 127 else 'white'  # Contrast color
                ax.text(c + GRID_THRESHOLD, r + GRID_THRESHOLD, "♕", color=queen_color,
                        fontsize=28, ha='center', va='center', fontweight='bold')
    ax.axis('off')
    plt.show()

""" Helper functions """

# Detect peaks in 1D data with normalization.
def detect_high_peaks(data, threshold=PEAK_THRESHOLD, min_distance=MIN_PEAK_DISTANCE):
    max_value = np.max(data)
    normalized_data = data / max_value
    peaks, _ = find_peaks(normalized_data, height=threshold, distance=min_distance)
    return peaks

# Check if two colors are within a threshold.
def within_threshold(color1, color2, threshold=COLOR_THRESHOLD):
    return all(abs(int(color1[i]) - int(color2[i])) <= threshold for i in range(3))

# Detect the size of the grid based on image analysis.
def detect_grid_size(image_array):
    gray_image = np.mean(image_array, axis=2).astype(np.uint8)
    row_sums = np.sum(255 - gray_image, axis=1)
    row_peaks = detect_high_peaks(row_sums, threshold=GRID_THRESHOLD, min_distance=10)
    global N
    N = len(row_peaks) + 1

# Extract the color of a specific cell.
def get_cell_color(image_array, row, col, cell_height, cell_width):
    upper_third_y = (row * cell_height) + (cell_height // 3)
    upper_third_x = (col * cell_width) + (cell_width // 3)
    return tuple(image_array[upper_third_y, upper_third_x])

# Check if a queen can be placed in a given cell.
def is_valid(board, regions, row, col, columns_used, regions_used):
    global N
    if columns_used[col] or regions[row][col] in regions_used:
        return False
    # No queens around current cell
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < N and 0 <= c < N and board[r][c] == "Q":
                return False
    return True

""" Solver functions """

# Solve the N-Queens problem using backtracking.
def solve_backtracking(board, regions, row, columns_used, regions_used):
    global N
    if row == N:
        return True
    for col in range(N):
        if is_valid(board, regions, row, col, columns_used, regions_used):
            board[row][col] = "Q"
            columns_used[col] = True
            regions_used.add(regions[row][col])
            if solve_backtracking(board, regions, row + 1, columns_used, regions_used):
                return True
            # Backtrack
            board[row][col] = "."
            columns_used[col] = False
            regions_used.remove(regions[row][col])
    return False

# Solve the N-Queens problem using integer linear programming.
def solve_ilp(board, regions):
    global N
    model = cp_model.CpModel()
    x = [[model.NewBoolVar(f"x[{i},{j}]") for j in range(N)] for i in range(N)]

    # One queen per row
    for i in range(N):
        model.Add(sum(x[i][j] for j in range(N)) == 1)
    # One queen per column
    for j in range(N):
        model.Add(sum(x[i][j] for i in range(N)) == 1)

    # No queens can be one block away from each other
    for i in range(N):
        for j in range(N):
            if i > 0:
                model.AddImplication(x[i][j], x[i - 1][j].Not())
            if i < N - 1:
                model.AddImplication(x[i][j], x[i + 1][j].Not())
            if j > 0:
                model.AddImplication(x[i][j], x[i][j - 1].Not())
            if j < N - 1:
                model.AddImplication(x[i][j], x[i][j + 1].Not())
            if i > 0 and j > 0:
                model.AddImplication(x[i][j], x[i - 1][j - 1].Not())
            if i > 0 and j < N - 1:
                model.AddImplication(x[i][j], x[i - 1][j + 1].Not())
            if i < N - 1 and j > 0:
                model.AddImplication(x[i][j], x[i + 1][j - 1].Not())
            if i < N - 1 and j < N - 1:
                model.AddImplication(x[i][j], x[i + 1][j + 1].Not())

    # Only one queen per region
    region_dict = {}
    for i in range(N):
        for j in range(N):
            region = regions[i][j]
            if region not in region_dict:
                region_dict[region] = []
            region_dict[region].append(x[i][j])
    for region, variables in region_dict.items():
        model.Add(sum(variables) <= 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL:
        for i in range(N):
            for j in range(N):
                if solver.Value(x[i][j]) == 1:
                    board[i][j] = "Q"
        return True
    return False

""" Main function """

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Solve and visualize the queens on a colored grid.")
    parser.add_argument("screenshot_name", help="Base name of the screenshot without file extension.")
    parser.add_argument("algorithm", nargs="?", default=None, help="Name of the algorithm used to solve the board (optional).")
    args = parser.parse_args()

    # Load the image
    image_path = f"boards/{args.screenshot_name}.png"
    image = Image.open(image_path)
    image_array = np.array(image)

    # Detect grid size
    detect_grid_size(image_array)
    height, width, _ = image_array.shape
    cell_height, cell_width = height // N, width // N

    # Extract grid colors
    grid_colors = [[get_cell_color(image_array, row, col, cell_height, cell_width) for col in range(N)] for row in range(N)]
    print_array(grid_colors)

    # Group colors into regions
    groups = {}
    regions = [[0 for _ in range(N)] for _ in range(N)]
    for r, row in enumerate(grid_colors):
        for c, color in enumerate(row):
            for group_color in groups:
                if within_threshold(color, group_color):
                    regions[r][c] = groups[group_color]
                    break
            else:
                groups[color] = len(groups) + 1
                regions[r][c] = groups[color]
    print_array(regions)

    # Solve the N-Queens problem
    start_time = time.time()
    board = [["." for _ in range(N)] for _ in range(N)]
    algorithm = args.algorithm
    solved = solve_ilp(board, regions) if algorithm == "ilp" else solve_backtracking(board, regions, 0, [False] * N, set())
    end_time = time.time()
    if solved:
        print_array(board)
        visualize_board(grid_colors, board)
        elapsed_time = end_time - start_time
        print(f"\nAlgorithmic time taken: {elapsed_time:.6f} seconds\n")
    else:
        print("\nNo solution found\n")

if __name__ == "__main__":
    main()
