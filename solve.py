import numpy as np
from PIL import Image
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

# parse command-line arguments
parser = argparse.ArgumentParser(description="Solve and visualize the queens on a colored grid.")
parser.add_argument("screenshot_name", help="Base name of the screenshot without file extension.")
args = parser.parse_args()
image_path = f"tests/{args.screenshot_name}.png"

# load the image
image = Image.open(image_path)

# outputs the boards
def print_array(array):
    print()
    for row in array:
        print(row)

# convert the image into a numpy array of RGB values
image_array = np.array(image)

# finds the size of the grid
def detect_grid_size(image_array):

    def detect_high_peaks(data, threshold=0.8, min_distance=10):
        max_value = np.max(data)
        normalized_data = data / max_value
        peaks, _ = find_peaks(normalized_data, height=threshold, distance=min_distance)
        return peaks

    gray_image = np.mean(image_array, axis=2).astype(np.uint8)
    row_sums = np.sum(255 - gray_image, axis=1)
    row_peaks = detect_high_peaks(row_sums, threshold=0.5, min_distance=10)
    return len(row_peaks) + 1

N = detect_grid_size(image_array)

# calculates the height and width of each cell in the grid
height, width, _ = image_array.shape
cell_height = height // N
cell_width = width // N

# extracts the color of the pixel at the upper 4th of each cell
def get_cell_color(image_array, row, col):
    upper_fourth_y = (row * cell_height) + (cell_height // 3)
    upper_fourth_x = (col * cell_width) + (cell_width // 3)
    color = image_array[upper_fourth_y, upper_fourth_x]
    return tuple(color)

# extract colors for all cells in the grid
grid_colors = [[get_cell_color(image_array, row, col) for col in range(N)] for row in range(N)]

print_array(grid_colors)

# check if two colors are within the threshold
def within_threshold(color1, color2, threshold = 20):
    return all(abs(int(color1[i]) - int(color2[i])) <= threshold for i in range(3))

# grouping colors
groups = {}
regions = [[0 for _ in range(N)] for _ in range(N)]
for rI, row in enumerate(grid_colors):
    for cI, color in enumerate(row):
        found = False
        for group_color in groups.keys():
            if within_threshold(color, group_color):
                regions[rI][cI] = groups[group_color]
                found = True
                break
        if not found:
            groups[color] = len(groups) + 1
            regions[rI][cI] = groups[color]

print_array(regions)

columns_used = [False] * N
regions_used = set()

# checks if a queen can be placed
def is_valid(board, row, col):
    if columns_used[col]:
        return False
    region = regions[row][col]
    if region in regions_used:
        return False
    # check for queens one tile away in all directions
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < N and 0 <= c < N and board[r][c] == "Q":
                return False
    return True

# solves the board using backtracking
def solve(board, row=0):
    # all queens placed
    if row == N:
        return True
    for col in range(N):
        if is_valid(board, row, col):
            # place queen
            board[row][col] = "Q"
            columns_used[col] = True
            regions_used.add(regions[row][col])
            # recurse to the next row
            if solve(board, row + 1):
                return True
            # backtrack if placing queen doesn't lead to a solution
            board[row][col] = "."
            columns_used[col] = False
            regions_used.remove(regions[row][col])
    return False

# visualize the board with colors
def visualize_board(grid_colors, board):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title("Linkedin Queens Solver")
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

board = [["." for _ in range(N)] for _ in range(N)]

if solve(board):
    # Visualize the result
    print_array(board)
    print()
    visualize_board(grid_colors, board)
else:
    print("\nNo solution found\n")
