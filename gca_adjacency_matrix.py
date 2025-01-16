import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse


def color_code_adjacency_matrix(graph_mat, state_mat):
    """
    Color-codes the adjacency matrix based on node types.
    """
    # Determine node types based on the state matrix
    node_types = [np.argmax(state_mat[i]) for i in range(state_mat.shape[0])]

    # Create a color-coded matrix for adjacency
    color_coded_mat = np.zeros_like(graph_mat, dtype=int)  # Initialize with 0 for no edge

    for i in range(graph_mat.shape[0]):
        for j in range(graph_mat.shape[1]):
            if graph_mat[i, j] == 1:  # Check if there is an edge
                type_i = node_types[i]
                type_j = node_types[j]

                # Assign colors based on type combinations
                if type_i == type_j:
                    # Same type, with specific colors for each cell type
                    if type_i == 0:
                        color_coded_mat[i, j] = 1  # Orange for same type 0
                    elif type_i == 1:
                        color_coded_mat[i, j] = 2  # Purple for same type 1
                    elif type_i == 2:
                        color_coded_mat[i, j] = 3  # Cyan for same type 2
                elif {type_i, type_j} == {1, 0}:
                    color_coded_mat[i, j] = 4  # Type 10 (blue)
                elif {type_i, type_j} == {1, 2}:
                    color_coded_mat[i, j] = 5  # Type 12 (green)
                elif {type_i, type_j} == {0, 2}:
                    color_coded_mat[i, j] = 6  # Type 02 (red)

    return color_coded_mat


def save_adjacency_plots(folder_path):
    """
    Generates and saves color-coded adjacency matrix plots with larger points.
    """
    adjacency_dir = os.path.join(folder_path, "adjacency_matrix")
    os.makedirs(adjacency_dir, exist_ok=True)

    # Create a fully opaque colormap
    cmap = ListedColormap([
        "white",  # No Edge
        "#FFA500",  # Orange for Same Type 0
        "#800080",  # Purple for Same Type 1
        "#00FFFF",  # Cyan for Same Type 2
        "#0000FF",  # Blue for Type 10
        "#008000",  # Green for Type 12
        "#FF0000",  # Red for Type 02
    ])

    # Process files in folder
    for filename in os.listdir(folder_path):
        if filename.endswith("_graph_mat.npy"):
            timepoint = filename.split("_")[0]

            # Load matrices
            graph_mat_path = os.path.join(folder_path, f"{timepoint}_graph_mat.npy")
            state_mat_path = os.path.join(folder_path, f"{timepoint}_state_mat.npy")

            # Ensure required files are present
            if not os.path.exists(state_mat_path):
                print(f"Skipping timepoint {timepoint}: Missing state matrix.")
                continue

            graph_mat = np.load(graph_mat_path)
            state_mat = np.load(state_mat_path)

            # Generate color-coded adjacency matrix
            color_coded_matrix = color_code_adjacency_matrix(graph_mat, state_mat)

            # Save the plot with larger points
            plt.figure(figsize=(10, 10))  # Increase figure size for larger points
            plt.title(f"Color-Coded Adjacency Matrix t = {timepoint}")
            plt.imshow(color_coded_matrix, cmap=cmap, interpolation="nearest")  # "nearest" makes points larger
            cbar = plt.colorbar(ticks=[0, 1, 2, 3, 4, 5, 6])
            cbar.ax.set_yticklabels([
                "No Edge", "Same Type 0 (Orange)", "Same Type 1 (Purple)", 
                "Same Type 2 (Cyan)", "Type 10 (Blue)", "Type 12 (Green)", "Type 02 (Red)"
            ])  # Custom labels
            plt.xlabel("Node Index")
            plt.ylabel("Node Index")
            plt.savefig(os.path.join(adjacency_dir, f"{timepoint}.png"))
            plt.close()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate color-coded adjacency matrix plots with larger points.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing input files.")

    args = parser.parse_args()
    input_folder = args.folder_path

    # Run the function
    if os.path.exists(input_folder):
        save_adjacency_plots(input_folder)
    else:
        print(f"Error: The folder '{input_folder}' does not exist.")
