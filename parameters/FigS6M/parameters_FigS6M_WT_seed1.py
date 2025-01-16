#
# Parameters for modeling the clustering of Pdgfra+ cells.
#

domain_size = [60, 14]   # domain width, height (number of cells)

# Thickness of top layer (now epithelium)
lumen_thickness = 6

# Mesenchymal layer thicknesses with corresponding Pdgfra- densities. 
# The three layers represent:
# 1: Pdgfra+/clustering layer, 
# 2-3: Bulk Pdgfra+ mesenchyme with variable number of Pdgfra+ cells (if required).
# If the sum of mesenchymal layers plus epithelial/lumen layer is less than domain
# height, the rest of the cells are assigned Pdgfra- state.
mes_thickness = [1, 0, 0]
mes_density = [0.0, 1.0, 1.0]

init_noise = 0.005      # initial arrangement noise
rng_seed = 1            # random number generator seed

dt = 0.025              # simulation time step
tMax = 2001             # simulation time end point

# Cell target areas for each type
A0 = [0.9, 0.9, 0.9]     # Pdgfra-, Pdgfra+, epithelium

# Cell target perimeter. Assigned as P0 = p0 * sqrt(A0), where p0 sets the 
# preferred shape as: 
# 3.722 -> hexagons
# 3.812 -> pentagons
# 4.0   -> squares
# 4.2426 -> rectangles with 1-2 proportions
# 4.559 -> triangles
# 4.6188 -> rectangles with 1-3 proportions
P0 = [3.812, 3.812, 3.812] # * np.sqrt(vor.A0)

# Motility
v0 = [0.75, 1.5, 0.0]       # Pdgfra-, Pdgfra+, epithelium

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

# Heterotypic adhesion energies
EM = 0.075      # Epithelium-Mesenchyme (s12 in Cahn-Hilliard)
EG = 0.075      # Epithelium-Green (s13)
MG = 0.100      # Mesenchyme-Green (s23)

# Homotypic adhesion energies (cohesion)
MM = 0.00
GG = 0.00
EE = 0.00

W = [MM, MG, EM], \
    [MG, GG, EG], \
    [EM, EG, EE]
