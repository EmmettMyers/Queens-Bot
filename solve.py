import numpy as np
from PIL import Image
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

# Constants
COLOR_THRESHOLD = 20
PEAK_THRESHOLD = 0.8
MIN_PEAK_DISTANCE = 10

# Utility Functions
def print_array(array):
    """Print a 2D array."""
    print()
    for row in array:
        print(row)
    print()

def detect_high_peaks(data, threshold=PEAK_THRESHOLD, min_distance=MIN_PEAK_DISTANCE):
    """Detect peaks in 1D data with normalization."""
    max_value = np.max(data)
    normalized_data = data / max_value
    peaks, _ = find_peaks(normalized_data, height=threshold, distance=min_distance)
    return peaks

def within_threshold(color1, color2, threshold=COLOR_THRESHOLD):
    """Check if two colors are within a threshold."""
    return all(abs(int(color1[i]) - int(color2[i])) <= threshold for i in range(3))

# Image Processing Functions
def detect_grid_size(image_array):
    """Detect the size of the grid based on image analysis."""
    gray_image = np.mean(image_array, axis=2).astype(np.uint8)
    row_sums = np.sum(255 - gray_image, axis=1)
    row_peaks = detect_high_peaks(row_sums, threshold=0.5, min_distance=10)
    return len(row_peaks) + 1

def get_cell_color(image_array, row, col, cell_height, cell_width):
    """Extract the color of a specific cell."""
    upper_third_y = (row * cell_height) + (cell_height // 3)
    upper_third_x = (col * cell_width) + (cell_width // 3)
    return tuple(image_array[upper_third_y, upper_third_x])

# Solver Functions
def is_valid(board, row, col, columns_used, regions_used, regions):
    """Check if a queen can be placed in a given cell."""
    if columns_used[col] or regions[row][col] in regions_used:
        return False

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < len(board) and 0 <= c < len(board) and board[r][c] == "Q":
                return False
    return True

def solve(board, row, columns_used, regions_used, regions):
    """Solve the N-Queens problem using backtracking."""
    if row == len(board):
        return True

    for col in range(len(board)):
        if is_valid(board, row, col, columns_used, regions_used, regions):
            board[row][col] = "Q"
            columns_used[col] = True
            regions_used.add(regions[row][col])

            if solve(board, row + 1, columns_used, regions_used, regions):
                return True

            board[row][col] = "."
            columns_used[col] = False
            regions_used.remove(regions[row][col])

    return False

# Visualization Function
def visualize_board(grid_colors, board):
    """Visualize the board with colors and queens."""
    N = len(board)
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title("LinkedIn Queens Solver")
    border_width = 0.1
    ax.add_patch(patches.Rectangle((-border_width, -border_width), N + 2 * border_width, N + 2 * border_width,
                                    facecolor='black', edgecolor='black', zorder=0))
    ax.set_xlim(-border_width, N + border_width)
    ax.set_ylim(-border_width, N + border_width)
    plt.gca().invert_yaxis()
    # colored tiles
    for r in range(N):
        for c in range(N):
            color = grid_colors[r][c]
            rect = patches.Rectangle((c, r), 1, 1, facecolor=np.array(color) / 255, edgecolor='black')
            ax.add_patch(rect)
    # queens visuals
    for r in range(N):
        for c in range(N):
            if board[r][c] == "Q":
                queen_color = 'black' if np.mean(grid_colors[r][c]) > 127 else 'white'  # Contrast color
                ax.text(c + 0.5, r + 0.5, "♕", color=queen_color,
                        fontsize=28, ha='center', va='center', fontweight='bold')
    ax.axis('off')
    plt.show()

# Main Execution
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Solve and visualize the queens on a colored grid.")
    parser.add_argument("screenshot_name", help="Base name of the screenshot without file extension.")
    args = parser.parse_args()

    # Load the image
    image_path = f"boards/{args.screenshot_name}.png"
    image = Image.open(image_path)
    image_array = np.array(image)

    # Detect grid size
    N = detect_grid_size(image_array)
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
    board = [["." for _ in range(N)] for _ in range(N)]
    columns_used = [False] * N
    regions_used = set()

    if solve(board, 0, columns_used, regions_used, regions):
        print_array(board)
        visualize_board(grid_colors, board)
    else:
        print("\nNo solution found\n")

if __name__ == "__main__":
    main()
