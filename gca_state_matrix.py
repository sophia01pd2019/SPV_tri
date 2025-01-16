import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse


def color_by_cell_type(state_mat):
    """
    Assign colors to nodes based on their cell type (dominant state).
    """
    node_colors = []
    for i in range(state_mat.shape[0]):
        cell_type = np.argmax(state_mat[i])  # Determine dominant state
        if cell_type == 0:
            node_colors.append("orange")  # Cell type 0
        elif cell_type == 1:
            node_colors.append("purple")  # Cell type 1
        elif cell_type == 2:
            node_colors.append("cyan")  # Cell type 2
    return node_colors


def save_state_graphs(folder_path):
    """
    Generates and saves bar plots for the state directory, color-coded by cell type,
    and includes a color bar for reference.
    """
    state_dir = os.path.join(folder_path, "states")
    os.makedirs(state_dir, exist_ok=True)

    # Process files in folder
    for filename in os.listdir(folder_path):
        if filename.endswith("_state_mat.npy"):
            timepoint = filename.split("_")[0]

            # Load state and properties matrices
            state_mat_path = os.path.join(folder_path, f"{timepoint}_state_mat.npy")
            properties_mat_path = os.path.join(folder_path, f"{timepoint}_properties_mat.npy")

            if not os.path.exists(properties_mat_path):
                print(f"Skipping timepoint {timepoint}: Missing properties matrix.")
                continue

            state_mat = np.load(state_mat_path)
            properties_mat = np.load(properties_mat_path)

            # Get colors based on cell types
            node_colors = color_by_cell_type(state_mat)

            # Generate and save bar plots
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2.5, 2.5]})

            # Plot first property (e.g., Cell Area)
            axes[0].bar(range(properties_mat.shape[0]), properties_mat[:, 0], color=node_colors)
            axes[0].set_title("Cell Area (Color-Coded by Cell Type)")
            axes[0].set_xlabel("Node Index")
            axes[0].set_ylabel("Area")

            # Plot second property (e.g., Cell Perimeter)
            axes[1].bar(range(properties_mat.shape[0]), properties_mat[:, 1], color=node_colors)
            axes[1].set_title("Cell Perimeter (Color-Coded by Cell Type)")
            axes[1].set_xlabel("Node Index")
            axes[1].set_ylabel("Perimeter")

            # Add a color bar for cell type reference
            cell_type_patches = [
                mpatches.Patch(color="orange", label="Cell Type 0"),
                mpatches.Patch(color="purple", label="Cell Type 1"),
                mpatches.Patch(color="cyan", label="Cell Type 2")
            ]
            fig.legend(handles=cell_type_patches, loc="center right", title="Cell Types", fontsize="small")

            plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to fit the legend
            plt.savefig(os.path.join(state_dir, f"{timepoint}.png"))  # Save with timepoint as filename
            plt.close()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate state folder graphs color-coded by cell type.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing input files.")

    args = parser.parse_args()
    input_folder = args.folder_path

    # Run the function
    if os.path.exists(input_folder):
        save_state_graphs(input_folder)
    else:
        print(f"Error: The folder '{input_folder}' does not exist.")
