"""
SPV simulation of intestinal cell clustering.

A 2D cross-sectional view of the intestinal tissue with three cell/material types: 
Pdgrfa- (dark green), Pdgfra+ (light green) and epithelium (red).

Run by calling this script and providing a parameters file as an arguments.
Writes output as npy data files into a given output folder and subfolders 
therein.

Notes: 
- If multithreading is not working in Ubuntu, try installing 
libopenblas0, libopenblas0-pthread
"""

from collections import defaultdict
import numpy as np
from scipy.stats import qmc
import time, os, shutil, sys 
# Local modules:
from tissue_pre_gca import *
#from tissue import *
import spv_stats, spv_mesh
import energy_heatmap as hm



def random_cell_states(rows, cols, ngfp, rng):
    """
    Generates a 2D array of randomly distributed cell states, 0 or 1, with given 
    total number of states 1. Positions of 1s are drawn from low discrepancy 
    Halton sequence.

    Parameters:
        rows: number of array rows
        cols: number of array columns
        ngfp: number of cells with state 1
        rng: random generator (np.random.detault_rng)

    Returns 
        Array of cell states, size rows x cols
    """

    # Array of cell identities with 0 = non-gfp, 1 = gfp.
    cells = np.zeros([rows,cols])
    if (ngfp == 0):
        return cells
    if (ngfp == rows*cols):
        return np.ones([rows,cols])
    
    # Generate low discrepancy distribution of gpf cells from the Halton sequence.
    # NOTE: qmc.Halton does not recognize the global np.random.seed().
    sampler = qmc.Halton(d=2, scramble=True, seed=rng)
    sample = sampler.random(n=ngfp)

    # Assign coordinates to cell nodes/bins, then set those cells to 1.
    # Note: Some coordinates may overlap; taking care of those below.
    i2d = np.floor(sample*[rows,cols])
    for i in i2d:
        cells[int(i[0]), int(i[1])] = 1
    
    # Inject the cells missing due to bin overlaps using the ordinary random numbers.
    n = ngfp - int(sum(sum(cells)))
    iz = np.where(cells == 0)
    inject = rng.integers(0, np.shape(iz)[1], size=n)
    # inject = np.random.randint(0, np.shape(iz)[1], size=n)   # same as above, the old way
    for i in inject:
        cells[iz[0][i], iz[1][i]] = 1
    
    return cells



#
# Handle input; read parameters into dictionary P.
#

if (len(sys.argv) < 2):
    print("Usage: python " + sys.argv[0] + " [parameters] [output folder (optional)]")
    exit(0)

sys.path.insert(0, os.path.dirname(sys.argv[1]))
s = os.path.basename(sys.argv[1])
par = __import__(s.split(".")[0])
P = defaultdict(lambda: [])
for p in dir(par):
    if (p[0:2] != "_"):
        P[p] = getattr(par, p)

# Some quick sanity testing.
if (len(P['domain_size']) != 2):
    print("Error: Parameter file has missing parameteres.")
    exit(1) 

#
# Initialize tissue; assign cell types.
#

# Initialize random generator. Used for generating random sequences throughout 
# this file, while tissue module will rely on its own generators.
rng = np.random.default_rng(P["rng_seed"])

m = int(P["domain_size"][1])        # initial number of cell rows
n = int(P["domain_size"][0])        # initial number of columns
m0 = int(P["lumen_thickness"])      # thickness of epithlium
m1 = int(P["mes_thickness"][0])     # Pdgfra+ 
m2 = int(P["mes_thickness"][1])     # Pdgfra- / shallow mesenchyme
m3 = int(P["mes_thickness"][2])     # Pdgfra- / deep mesenchyme
m4 = m - m0 - m1 - m2 - m3          # Pure Pdgfra- / bottom gap 

x0 = spv_mesh.init_square_lattice(rows=m, cols=n, rng=rng, noise=P["init_noise"])
print(np.shape(x0))
vor = Tissue(x0, P)

# Construct GFP layers in top to bottom order, with top layer(s) having the 
# highest density.
gfp = []
ngfp = m1*(1.0-P["mes_density"][0])
for i in range(0,m1):
    ngfp = np.max([0.0, ngfp])
    p = np.min([ngfp, 1.0])
    arr = np.array( [1]*int(p*n) + [0]*(n-int(p*n)) )
    rng.shuffle(arr)
    # np.random.shuffle(arr)    # old style shuffle
    gfp = np.concatenate([gfp, arr])
    ngfp = ngfp - 1.0

gfp = np.flip(gfp)

# If Pdgfra- mesenchyme density (mes_density) is less than 1, the mensenchyme is
# injected with Pdgfra+ cells to simulate SAG treatment effects resulting in deep
# clustering. Following ptch1 data, moving deeper into the mesenchyme there is 
# first Pdgfra+ cells at lower density, then Pdgfra+ cells increasing in density.
# Finally, to avoid periodicity effects (clustering cells adhering to the top of
# the epithelium) in the current simulations, a thin gap with no Pdgfra+ cells is
# set as the final layer.
# l2 = round(0.5*(m2-m3))   # nunber of rows, top layer
# l1 = (m2-m3) - l2         # number of rows, bottom layer

# First layer (bottom)
mes1 = round(m3 * n * (1.0-P["mes_density"][2]))  # number of Pdgfra+ cells
mes0 = m3*n - mes1                              # number of Pdgfra- cells
mm1 = random_cell_states(m3, n, mes1, rng=rng)
print("Bottom mesenchymal layer, number of Pdgfra low and high cells: " \
      + str(mes0) + ", " + str(mes1))

# Second layer (the one touching epithelium)
mes1 = round(m2 * n * (1.0-P["mes_density"][1]))
mm2 = random_cell_states(m2, n, mes1, rng=rng)
print("Top mesenchymal layer, number of Pdgfra low and high cells: " \
      + str(mes0) + ", " + str(mes1))

# Combine layers 1 and 2, add bottom gap.
mes = np.concatenate([np.zeros(int(m4*n)), mm1.ravel(), mm2.ravel()])

# Set the final cell type array.
c_types = [ mes, \
            gfp, \
            np.full(m0*n, 2) ]
c_types = np.concatenate(c_types)
c_types = c_types.astype(int)

#
# Set simulation settings, execute simulation.
#

kE = par.W[0][2] + par.W[1][2] - par.W[0][1] 
kG = par.W[0][1] + par.W[1][2] - par.W[0][2] 
kM = par.W[0][1] + par.W[0][2] - par.W[1][2]
print("Effective tissue cohesions: kE=%f, kG=%f, kM=%f" %(kE, kG, kM))

vor.set_interaction( W=np.asarray(par.W), c_types=c_types, pE=[], randomize=False )

# Output folder for writing results.
s = os.path.basename(sys.argv[1])
tokens = s.split(".")[0].split("_")
if "parameters" in tokens:
    tokens.remove("parameters")

prefix = '_'.join(tokens)
outputDir = ""
if len(sys.argv) == 3:
    outputDir = sys.argv[2] + "/"
outputDir = outputDir + "%s_%d" %(prefix, time.time())
# outputDir = "%s_%d_size-%d_activ-%s" %(prefix, time.time(), par.domain_size[0], \
#             str(vor.v0).replace(" ", ""))
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# Save this script, parameters to results.
shutil.copy2(__file__, outputDir)
shutil.copy2(sys.argv[1], outputDir)

# Set simulation time step, end time.
vor.set_t_span(par.dt, par.tMax)

# Print_every sets the number of iterations for which to print the percentage completed.
t_start = time.time()
x_save, tri_save = vor.simulate( print_every=100, output_dir=outputDir, rng_seed=P["rng_seed"] )
print('Simulation took %f seconds' %(time.time() - t_start))

#
# Write additional statistics, plots.
#

print('Writing additional data/statistics...')
t_start = time.time()

spv_stats.write_velocity_stats( x_save, c_types, par.dt, output_file=outputDir+"/stats_velocity.txt" )
spv_stats.write_connection_matrix( tri_save[-1], c_types, output_folder=outputDir )

# Plot cell-cell interaction energies matrix as a heatmap.
'''
fig, ax = plt.subplots(dpi=200)
im = hm.heatmap(np.asarray(par.W), vor.colors, vor.colors, ax=ax, cmap="YlGn")
texts = annotate_heatmap(im)
fig.tight_layout()
plt.savefig(outputDir + "/cell_interaction_energies.png")
'''

print('done after %f seconds' %(time.time() - t_start))
