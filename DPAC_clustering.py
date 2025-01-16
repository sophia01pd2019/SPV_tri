"""
SPV simulation of cell clustering with periodic boundaries and two material phases:
1. Substrate (phase 1) covering the whole domain by default.
2. Cells (phase 2) introduced as a stripe as a horizontal stripe with given 
thickness and density.

Parameters:
- stripe_thickness: Number of rows in the central stripe.
- stripe_density: Fraction of substrate in the stripe flipped to cells.

Writes output as npy data files into a given output folder.
"""

from collections import defaultdict
import numpy as np
import time, os, shutil, sys 
# Local modules:
from tissue import *
import spv_stats, spv_mesh


def initialize_phases(rows, cols, stripe_thickness, stripe_density, rng):
    """
    Initializes the domain with two phases:
    - Phase 1 (substrate): Covers the entire domain initially.
    - Phase 2 (cells): Introduced in a horizontal central stripe.

    Parameters:
        rows: Number of rows in the domain.
        cols: Number of columns in the domain.
        stripe_thickness: Number of rows in the stripe.
        stripe_density: Fraction of substrate flipped to cells in the stripe.
        rng: Random number generator.

    Returns:
        Array of phase types (0 for substrate, 1 for cells).
    """
    # Initialize entire domain as substrate (phase 1).
    c_types = np.zeros((rows, cols), dtype=int)

    # Calculate stripe start and end rows.
    stripe_start = (rows - stripe_thickness) // 2
    stripe_end = stripe_start + stripe_thickness

    # Flip a fraction of cells in the stripe to phase 2 (cells).
    for row in range(stripe_start, stripe_end):
        flip_indices = rng.choice(
            cols, size=int(stripe_density * cols), replace=False
        )
        c_types[row, flip_indices] = 1

    return c_types.ravel()


#
# Handle input; read parameters into dictionary P.
#
if len(sys.argv) < 2:
    print("Usage: python " + sys.argv[0] + " [parameters] [output folder (optional)]")
    exit(0)

sys.path.insert(0, os.path.dirname(sys.argv[1]))
s = os.path.basename(sys.argv[1])
par = __import__(s.split(".")[0])
P = defaultdict(lambda: [])
for p in dir(par):
    if not p.startswith("_"):
        P[p] = getattr(par, p)

if len(P['domain_size']) != 2:
    print("Error: Parameter file has missing parameters.")
    exit(1)

# Initialize random generator
rng = np.random.default_rng(P["rng_seed"])

m = int(P["domain_size"][1])                    # number of cell rows
n = int(P["domain_size"][0])                    # number of cell columns
stripe_thickness = int(P["stripe_thickness"])   # thickness of the central stripe
stripe_density = P["stripe_density"]            # density of cells in the stripe

x0 = spv_mesh.init_square_lattice(rows=m, cols=n, rng=rng, noise=P["init_noise"])
vor = Tissue(x0, P)

# Generate phase types: substrate and cells
c_types = initialize_phases(m, n, stripe_thickness, stripe_density, rng=rng)

# Set interactions
vor.set_interaction(W=np.asarray(par.W), c_types=c_types, pE=[], randomize=False)

# Output folder
s = os.path.basename(sys.argv[1])
outputDir = f"{'_'.join(s.split('.')[0].split('_'))}_{int(time.time())}"
if len(sys.argv) == 3:
    outputDir = os.path.join(sys.argv[2], outputDir)
os.makedirs(outputDir, exist_ok=True)

shutil.copy2(__file__, outputDir)
shutil.copy2(sys.argv[1], outputDir)

# Simulation
vor.set_t_span(par.dt, par.tMax)
t_start = time.time()
x_save, tri_save = vor.simulate(print_every=100, output_dir=outputDir, rng_seed=P["rng_seed"])
print('Simulation completed in %f seconds' % (time.time() - t_start))

# Statistics
spv_stats.write_velocity_stats(x_save, c_types, par.dt, output_file=os.path.join(outputDir, "stats_velocity.txt"))
spv_stats.write_connection_matrix(tri_save[-1], c_types, output_folder=outputDir)

print('Simulation data saved successfully.')
