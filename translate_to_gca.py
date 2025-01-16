import numpy as np
import os
from scipy.sparse import csr_matrix

def npy_to_gca(file_path):
    """
    Translates a single .npy file into DGCA-compatible matrices.
    
    :param file_path: Path to the .npy file.
    :return: A tuple (graph_mat, state_mat, properties_mat).
    """
    # Load the .npy file
    data = np.load(file_path, allow_pickle=True)
    domain_size, x_save, c_types, tri_save, area, perimeter = data

    # 1. Construct the adjacency matrix (graph_mat) based on triangulation
    n_cells = len(c_types)
    adjacency = np.zeros((n_cells, n_cells), dtype=int)
    for tri in tri_save:
        for i in range(3):
            adjacency[tri[i], tri[(i + 1) % 3]] = 1
            adjacency[tri[(i + 1) % 3], tri[i]] = 1
    graph_mat = csr_matrix(adjacency)  # Sparse representation

    # 2. Construct the state matrix (state_mat)
    n_cell_types = len(np.unique(c_types))
    state_mat = np.zeros((n_cells, n_cell_types))
    for i, cell_type in enumerate(c_types):
        state_mat[i, cell_type] = 1  # One-hot encoding of cell types

    # 3. Construct the properties matrix (properties_mat)
    properties_mat = np.column_stack((area, perimeter))

    return graph_mat, state_mat, properties_mat


def process_npy_folder(input_folder, output_folder):
    """
    Processes all .npy files in a folder and converts them to DGCA matrices.
    
    :param input_folder: Folder containing .npy simulation files.
    :param output_folder: Folder to save the converted DGCA matrices.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]

    for i, file_name in enumerate(npy_files):
        print(f"Processing file {i + 1}/{len(npy_files)}: {file_name}")
        file_path = os.path.join(input_folder, file_name)

        # Convert to DGCA matrices
        graph_mat, state_mat, properties_mat = npy_to_gca(file_path)

        # Save results
        base_name = os.path.splitext(file_name)[0]
        np.save(os.path.join(output_folder, f"{base_name}_graph_mat.npy"), graph_mat.toarray())
        np.save(os.path.join(output_folder, f"{base_name}_state_mat.npy"), state_mat)
        np.save(os.path.join(output_folder, f"{base_name}_properties_mat.npy"), properties_mat)

    print(f"Processing complete. Results saved in {output_folder}")


if __name__ == "__main__":
    import argparse

    # Argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Translate .npy files to DGCA matrices.")
    parser.add_argument("input_folder", type=str, help="Folder containing .npy simulation files")
    parser.add_argument("output_folder", type=str, help="Folder to save converted DGCA matrices")

    args = parser.parse_args()

    # Process the folder
    process_npy_folder(args.input_folder, args.output_folder)
