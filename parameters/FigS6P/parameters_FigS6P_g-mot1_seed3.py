#
# Parameters for modeling the clustering of Pdgfra+ cells.
#

domain_size = [60, 14]   # domain width, height (number of cells)

# Thickness of top layer (now epithelium)
lumen_thickness = 6    

# Mesenchymal layer thicknesses with corresponding Pdgfra- densities. 
# The three layers represent:
# 1: Pdgfra+/clustering layer, 
# 2-3: Bulk Pdgfra- mesenchyme with variable number of Pdgfra+ cells (if required).
# If the sum of mesenchymal layers plus epithelial/lumen layer is less than domain
# height, the rest of the cells are assigned Pdgfra- state.
mes_thickness = [1, 0, 0]
mes_density = [0.0, 1.0, 1.0]

init_noise = 0.005      # initial arrangement noise
rng_seed = 3            # random number generator seed

dt = 0.025              # simulation time step
tMax = 2001             # simulation time end point

# Cell target areas for each type
# A0 = [0.9, 0.9, 0.9, 0.9]
A0 = [0.9, 0.9, 0.9]    # red, green, white

# Cell target perimeter. Assigned as P0 = p0 * sqrt(A0), where p0 sets the 
# preferred shape as: 
# 3.722 -> hexagons
# 3.812 -> pentagons
# 4.0   -> squares
# 4.2426 -> rectangles with 1-2 proportions
# 4.559 -> triangles
# 4.6188 -> rectangles with 1-3 proportions
P0 = [3.812, 3.812, 3.812] # * np.sqrt(vor.A0)
# P0 = [3.812, 3.812, 3.812, 3.812]

# Motility
v0 = [0.75, 1.0, 0.0]      # for red, green, lumen

# Inverse of persistence timescale for motility
Dr = 40

# Coefficient for deviation from target area in energy functional
kappa_A = 0.3

# Same as above for perimeter
kappa_P = 0.05

# Maximal radius to engage in repulsion
a = 0.2   

# Coefficient of repulsion.
k = 2     

# Adhesion energies between cell types
RG = 0.10      # red-green
CX = 0.075       # white-X (lumen)
CC = 0.00

RR = 0.00       # red-red
GG = 0.00       # green-green

W = [RR, RG, CX], \
    [RG, GG, CX], \
    [CX, CX, CC]

# W3 *= 0.1 # * vor.P0 / r    # TODO: justify scaling by vor.P0/r
