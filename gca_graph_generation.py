import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


def plot_graph(graph_mat, state_mat, output_path):
    """
    Visualizes the adjacency graph with node colors and opacity based on state matrix.
    Saves the output as a PNG file.
    """
    # Create a graph from adjacency matrix
    g = nx.from_numpy_array(graph_mat, create_using=nx.Graph)

    # Node colors and opacity based on state
    node_colors = []
    node_opacity = []
    for i in range(state_mat.shape[0]):
        cell_type = np.argmax(state_mat[i])  # Get the dominant state
        if cell_type == 0:  # Cell type 0
            node_colors.append("yellow")
            node_opacity.append(0.5)  # 50% opacity
        elif cell_type == 1:  # Cell type 1
            node_colors.append("purple")
            node_opacity.append(1.0)  # 100% opacity
        elif cell_type == 2:  # Cell type 2
            node_colors.append("green")
            node_opacity.append(0.5)  # 50% opacity

    # Create a figure with dedicated axes
    fig, ax = plt.subplots(figsize=(8, 8))
    pos = nx.kamada_kawai_layout(g)  # Use Kamada-Kawai layout for visualization

    # Draw the graph with specified opacity
    nodes = nx.draw_networkx_nodes(
        g,
        pos,
        node_color=node_colors,
        alpha=node_opacity,  # Apply varying opacities
        node_size=50,
        ax=ax
    )
    nx.draw_networkx_edges(g, pos, edge_color="gray", ax=ax)

    # Add a legend instead of a colorbar
    cell_type_patches = [
        mpatches.Patch(color="yellow", label="Cell Type 0"),
        mpatches.Patch(color="purple", label="Cell Type 1"),
        mpatches.Patch(color="green", label="Cell Type 2")
    ]
    ax.legend(handles=cell_type_patches, loc="upper right", title="Cell Types", fontsize="small")

    plt.title("Graph Representation (Kamada-Kawai Layout with Opacity)")
    plt.savefig(output_path)
    plt.close()


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


def save_state_graphs(folder_path, state_dir, state_mat, properties_mat, timepoint):
    """
    Generates and saves bar plots for the state directory, color-coded by cell type,
    with a legend matching the graph representation color scheme.
    """
    # Get colors based on cell types
    node_colors = []
    for i in range(state_mat.shape[0]):
        cell_type = np.argmax(state_mat[i])  # Determine dominant state
        if cell_type == 0:
            node_colors.append("yellow")  # Cell type 0
        elif cell_type == 1:
            node_colors.append("purple")  # Cell type 1
        elif cell_type == 2:
            node_colors.append("green")  # Cell type 2

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

    # Add a legend matching the graph representation color scheme
    cell_type_patches = [
        mpatches.Patch(color="yellow", label="Cell Type 0"),
        mpatches.Patch(color="purple", label="Cell Type 1"),
        mpatches.Patch(color="green", label="Cell Type 2")
    ]
    fig.legend(handles=cell_type_patches, loc="center right", title="Cell Types", fontsize="small")

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to fit the legend
    plt.savefig(os.path.join(state_dir, f"{timepoint}.png"))  # Save with timepoint as filename
    plt.close()


def visualize_and_save(folder_path):
    """
    Processes all timepoint files in the given folder and saves visualizations
    in respective directories.
    """
    # Create output directories
    adjacency_dir = os.path.join(folder_path, "adjacency_matrix")
    graph_dir = os.path.join(folder_path, "graph_representation")
    state_dir = os.path.join(folder_path, "states")

    os.makedirs(adjacency_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)

    # Process files in folder
    for filename in os.listdir(folder_path):
        if filename.endswith("_graph_mat.npy"):
            timepoint = filename.split("_")[0]

            # Load matrices
            graph_mat_path = os.path.join(folder_path, f"{timepoint}_graph_mat.npy")
            state_mat_path = os.path.join(folder_path, f"{timepoint}_state_mat.npy")
            properties_mat_path = os.path.join(folder_path, f"{timepoint}_properties_mat.npy")

            if not (os.path.exists(state_mat_path) and os.path.exists(properties_mat_path)):
                print(f"Skipping timepoint {timepoint}: Missing required files.")
                continue

            graph_mat = np.load(graph_mat_path)
            state_mat = np.load(state_mat_path)
            properties_mat = np.load(properties_mat_path)

            # Save color-coded adjacency matrix
            color_coded_matrix = color_code_adjacency_matrix(graph_mat, state_mat)
            cmap = ListedColormap([
                "white",  # No Edge
                "#FFA500",  # Orange for Same Type 0
                "#800080",  # Purple for Same Type 1
                "#00FFFF",  # Cyan for Same Type 2
                "#0000FF",  # Blue for Type 10
                "#008000",  # Green for Type 12
                "#FF0000",  # Red for Type 02
            ])
            plt.figure(figsize=(10, 10))
            plt.title(f"Color-Coded Adjacency Matrix t = {timepoint}")
            plt.imshow(color_coded_matrix, cmap=cmap, interpolation="nearest")
            cbar = plt.colorbar(ticks=[0, 1, 2, 3, 4, 5, 6])
            cbar.ax.set_yticklabels([
                "No Edge", "Same Type 0 (Orange)", "Same Type 1 (Purple)",
                "Same Type 2 (Cyan)", "Type 10 (Blue)", "Type 12 (Green)", "Type 02 (Red)"
            ])
            plt.xlabel("Node Index")
            plt.ylabel("Node Index")
            plt.savefig(os.path.join(adjacency_dir, f"{timepoint}.png"))
            plt.close()

            # Save graph representation visualization
            plot_graph(graph_mat, state_mat, os.path.join(graph_dir, f"{timepoint}.png"))

            # Save state directory visualizations
            save_state_graphs(folder_path, state_dir, state_mat, properties_mat, timepoint)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process and visualize matrices for graph analysis.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing input files.")

    args = parser.parse_args()
    input_folder = args.folder_path

    if os.path.exists(input_folder):
        visualize_and_save(input_folder)
    else:
        print(f"Error: The folder '{input_folder}' does not exist.")
