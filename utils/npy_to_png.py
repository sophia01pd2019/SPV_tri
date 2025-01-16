"""
Reads simulation state from the given .npy file and writes a colored png image
with the same file body name.

Currently assumes three phases/cell types (see TODO below).
"""

import numpy as np
import sys, glob
import pathlib
# Assume SPV_plot is one up relative to this script:
sys.path.append( str(pathlib.Path(__file__).parent.resolve()) + "/../" )
from spv_plot import *

# Width of the polygon edge line.
line_width = 0.5

# Colors to use for three phases.
colors = "#004D40", "#02E002", "#9E039F"      # dark green, light green, red/purple 
# colors = "#004D40", "#02E002", "#D81B60"    # dark green, light green, red
# colors = "#004D40", "#65FF65", "#D81B60"    # dark green, bright-light green, red
# colors = "#004D40", "#FFC107", "#D81B60"    # dark green, yellow, red
# colors = "#1E88E5", "#004D40", "#D81B60"    # blue, dark green, red
# colors = "#1E88E5", "#FFC107", "#D81B60"    # blue, yellow, red

# If input file given, use that. Otherwise, take all files in the folder.
if len(sys.argv) == 2:
    input_files = [sys.argv[1]]
else:
    input_files = glob.glob("*.npy")

if not input_files:
    print("No .npy files found.")
    sys.exit(1)

for i, file in enumerate(input_files):
    print(f"Processing file {i + 1} / {len(input_files)}: {file}")

    # Read domain dimensions (L), vertex positions (X), cell types (c_types), and triangulation (tri_save).
    try:
        L, X, c_types, tri_save = np.load(file, allow_pickle=True)
    except ValueError:
        print(f"File {file} does not match expected format. Skipping.")
        continue

    output = file.split(".")[0]  # output file name without .png extension

    # Generate PNG using `plot_step`, passing triangulation
    plot_step(
        data=X,
        iter=output,
        domain_size=L,
        c_types=c_types,
        colors=colors,
        plot_scatter=True,
        tri_save=tri_save,
        dir_name="output_images",  # Save images in this directory
        line_width=line_width,
        an_type="periodic",
    )

print("PNG generation complete.")
