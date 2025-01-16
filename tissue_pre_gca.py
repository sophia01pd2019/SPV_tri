"""Representation of self-propelled Voronoi tissue.

Implements self-propelled Voronoi model based on methods described in:
- Barton et al. Active Vertex Model for cell-resolution description of epithelial 
tissue mechanics (2017).
- Bi et al. Motility-Driven Glass and Jamming Transitions in Biological Tissues
(2016).

Fork of the implementation by Jake Cornwall-Scoones. Original project source:
https://github.com/jakesorel/active_vertex/

For usage, see gut_SPV.py.

TODO: 
- Improve documentation, clean-up.

"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time, os, math
from scipy.spatial import Delaunay
# from line_profiler import LineProfiler
from matplotlib import cm
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
# Local modules:
from spv_math import *
from spv_plot import *


class Tissue:
    def __init__(self, x0=[], P=[]):
        """
        Initializes the tissue.

        :param x0: initial cell coordinates
        :param P: model parameters (as a dictionary)
        """
        # if (x0 == []):
        #     x0 = np.array([])
        if (P == []):
            P = defaultdict(lambda: [])
            # Assign default values other than '[]' if desired:
            P['A0'] = 1.0
            P['P0'] = 3.4641
            P['kappa_A'] = 1
            P['kappa_P'] = 1

        self.n_c = x0.shape[0]      # number of cells (Delaunay vertices)
        self.n_C = self.n_c         # ??
        self.n_v = []               # number of Voronoi vertices
        self.x0 = x0                # initial cell coordinates
        self.x = self.x0            # cell coordinates (updated)
        self.vs = []                # Voronoi vertex coordinates
        self.tris = []              # Delaunay triangles !!!
        self.Cents = []             # node coordinates for each triangle; x[tris]
        self.v0 = P['v0']           # Instatantaneous velocity (activity)
        self.Dr = P['Dr']           # Inverse of persistence timescale for motility

        # For these, see the Active Vertex PLOS paper (Barton et al. 2017). 
        # At very short distances cells show spring-like repulsion.
        self.a = P['a']                         # Maximal radius to engage in repulsion
        self.k = P['k']                         # Coefficient of repulsion.

        self.A0 = P['A0']                       # reference (initial) area
        self.P0 = P['P0'] * np.sqrt(self.A0)    # reference (initial) perimeter
        self.kappa_A = P['kappa_A']             # area modulus (resistance to changing cell area)
        self.kappa_P = P['kappa_P']             # perimeter modulus

        self.J = []                             # cell-cell interactions for all cells
        self.c_types = []                       # cell type/category for all cells
        self.domain_size = np.asarray(P['domain_size']) # domain size (width, height)
        self.L = 0                              # TODO: get rid of L, use domain_size instead

        self.k2s = []                           # ??

        # TODO: instead of storing these, write output at every iteration
        self.x_save = []                        # cell centroids for each time step
        self.tri_save = []                      # triangulation for each time step

        self.colors = (1,0,0,0.5),(0,0,1,0.5)   # color for each cell type/category
        self.plot_scatter = False               # plot dots at the cell centroids
        self.plot_forces = False                # overlay a quiver plot of the forces
        self.cell_movement_mask = None          # ??
        self.no_noise_time = None               # ??
        self.b_extra = 2                        # ??
        self.noise = []                         # TODO: locally return from noise generator



    def hexagonal_lattice(self, rows=3, cols=3, noise=0.0005, A=None):
        """
        Assemble a hexagonal lattice

        :param rows: Number of rows in lattice
        :param cols: Number of columns in lattice
        :param noise: Noise added to cell locs (Gaussian SD)
        :return: points (nc x 2) cell coordinates.
        """
        points = []
        for row in range(rows * 2):
            for col in range(cols):
                x = (col + (0.5 * (row % 2))) * np.sqrt(3)
                y = row * 0.5
                x += np.random.normal(0,noise)
                y += np.random.normal(0,noise)
                points.append((x, y))
        
        points = np.asarray(points)
        if A is not None:
            points = points * np.sqrt(2*np.sqrt(3)/3)
        
        return points



    def make_init(self, domain_size, noise=0.005, rng_seed=1):
        """
        Make initial condition. Currently, this is a hexagonal lattice + noise.

        Makes reference to the self.hexagonal_lattice function, then crops down 
        to the reference frame

        Stores:
            self.n_c = number of cells
            self.x0 = (nc x 2) matrix denoting cell coordinates
            self.x = clone of self.x0

        :param domain_size: Domain width and height (np.float32, np.float32)
        :param rng_seed: Initialized the random number generator (int32)
        :param noise: Gaussian noise added to {x,y} coordinates (np.float32)
        """

        np.random.seed(rng_seed)
        domain_size = np.asarray(domain_size)
        x0 = self.hexagonal_lattice( int(np.ceil(domain_size[1]/0.5)), \
                                     int(np.ceil(domain_size[0]/np.sqrt(3))), \
                                     noise=noise )
        # self.x0 = self.hexagonal_lattice(self.n_c,self.n_c,noise=noise)
        # self.x0 = self.x0[self.x0.max(axis=1) < L*0.95]

        # TODO: Clean up/document the hardcoded constants.
        x0 += 1e-3
        # With cols=rows=[number of cells]=n this generates a field with n*(n+1) cells.        
        x0 = x0[x0[:,0] < domain_size[0]*0.97]
        x0 = x0[x0[:,1] < domain_size[1]*0.97]

        self.x0 = x0
        self.x = x0
        self.n_c = x0.shape[0]
        self.n_C = self.n_c
        self.domain_size = domain_size
        self.L = domain_size[0]     # TODO: get rid of L, use domain_size instead



    def make_init_boundary(self, L, r, noise=0.005, rng_seed=1):
        self.make_init([L,L], noise=noise, rng_seed=rng_seed)
        self._triangulate(self.x0)
        circular_mask = (self.x0[:,0] - self.L/2)**2 + (self.x0[:,1] - self.L/2)**2 <= (r*L)**2
        neighs = []

        for i, tri in enumerate(self.tris):
            In = 0
            for c in tri:
                if circular_mask[c]:
                    In +=1
            if In !=0:
                for c in tri:
                    neighs.append(c)
        
        kept = list(np.nonzero(circular_mask)[0])
        boundary_particles = list(set(neighs).difference(set(list(kept))))

        self.x = self.x[kept+boundary_particles]
        self.x0 = self.x.copy()
        self.n_c = self.x0.shape[0]
        self.n_C = len(kept)
        self.domain_size = np.array([L, L])



    def set_interaction(self, W = 0.16*np.array([[2, 0.5], [0.5, 2]]), \
                        pE = 0.5, c_types=None, randomize=True):
        """
        Set cell-cell interaction matrix W with n columns and row for n cell
        types. Element (i,j) of W is the interaction between cells i and j.
        W is assumed to be symmetric.

        :param W: interactions matrix
        :param pE: Fraction of cells belonging to class E.
        :param c_types: ...
        :param randomize: ...
        """
        print("Setting interactions for %d cells." %self.n_C)

        if c_types is None:
            nE = int(self.n_C * pE)   # number of cells type E
            N_dict = {"E": nE, "T": self.n_C - nE,}

            c_types = np.zeros(self.n_C, dtype=np.int32)
            j = 0
            for k, c_type in enumerate(N_dict):
                j1 = N_dict[c_type]
                c_types[j:j + j1] = k
                j += j1
            if randomize is True:
                np.random.shuffle(c_types)

        if self.n_c != self.n_C:
            c_types = np.concatenate( (c_types, np.repeat(-1,self.n_c-self.n_C)) )

        # Contruct interactions matrix for all cell-cell pairs.
        cell_i, cell_j = np.meshgrid( c_types, c_types, indexing="ij" )
        J = W[cell_i, cell_j]
        self.J = J
        self.c_types = c_types



    def set_interaction_boundary(self, W = 0.16*np.array([[2, 0.5], [0.5, 2]]), \
                                 pE = 0.5, Wb = [0.16,0.16], b_extra = 3):
        self.b_extra = b_extra

        nE = int(self.n_C*pE)
        N_dict = {"E": nE, "T": self.n_C - nE}

        c_types = np.zeros(self.n_C, dtype=np.int32)
        j = 0
        for k, c_type in enumerate(N_dict):
            j1 = N_dict[c_type]
            c_types[j:j + j1] = k
            j += j1
        np.random.shuffle(c_types)

        c_types_all = np.concatenate((c_types,np.repeat(-1,self.n_c-self.n_C)))

        cell_i, cell_j = np.meshgrid(c_types, c_types, indexing="ij")
        J = W[cell_i, cell_j]
        self.J = J
        self.c_types = c_types
        self.c_types_all = c_types_all

        # Save a larger matrix from which to sample. This defines the interaction 
        # strengths of cells with the boundary
        self.J_large = np.zeros((self.n_c*self.b_extra,self.n_c*self.b_extra))
        for i, c_type in enumerate(self.c_types):
            self.J_large[i] = Wb[c_type]
            self.J_large[:,i] = Wb[c_type]
        self.J_large[:self.n_C,:self.n_C] = self.J



    def assign_vertices(self):
        """
        Generate the CV_matrix, an (nc x nv x 3) array, considering the relationship 
        between cells and vertices/triangulation. Essentially an array expression 
        of the triangulation.

        Row i of self.tris contains the nodes of the ith triangle, with each node
        corresponding to a cell (total n_c cells). Each triangle encloses a Voronoi 
        vertex (total n_v vertices).

        TODO: Replace with a simpler cell id -> Voronoi nodes -map, implement 
              in the generic mesh class.

        Uses the stored self.tris, the (nv x 3) array denoting the triangulation.

        :return self.CV_matrix: array representation of the triangulation (nc x nv x 3)
        """
        CV_matrix = np.zeros((self.n_c, self.n_v, 3))
        for i in range(3):
            CV_matrix[self.tris[:, i], np.arange(self.n_v), i] = 1
        
        self.CV_matrix = CV_matrix
        
        return self.CV_matrix



    def set_t_span(self, dt, tfin):
        """
        Set the temporal running parameters

        :param dt: Time-step (np.float32)
        :param tfin: Final time-step (np.float32)
        :return self.t_span: Vector of times considered in the simulation (nt x 1)
        """
        self.dt, self.tfin = dt,tfin
        self.t_span = np.arange(0,tfin,dt)

        return self.t_span



    def generate_noise(self, rng_seed=1):
        """
        Generates random motility noise for all cells and time steps as 
        (n_t, n_c, 2) array for n_t time steps, n_c cells and 2 spatial dimensions.
        
        Implemented as random rotational diffusion with persistence(?)
        """

        if type(self.noise) is list:
            # Generating normal and uniform components of the noise separately; 
            # re-apply RNG seed in-between to allow for deterministic behavior.
            # TODO: Document the rationale here/cite something.
            np.random.seed(rng_seed)
            noise_normal = np.random.normal(0, np.sqrt(2 * self.Dr * self.dt), (self.n_t, self.n_c))
            np.random.seed(rng_seed)
            noise_uniform = np.random.uniform(0, np.pi*2, self.n_c)

            theta_noise = np.cumsum(noise_normal, axis=0) + noise_uniform
            self.noise = np.dstack((np.cos(theta_noise), np.sin(theta_noise)))

            if self.cell_movement_mask is not None:
                self.noise[:,~self.cell_movement_mask] = self.noise[:,~self.cell_movement_mask]*0
            
            if self.no_noise_time is not None:
                self.noise[:self.no_noise_time] = 0*self.noise[:self.no_noise_time]



    def generate_noise_boundary( self, rng_seed=1 ):
        n_c_extra = int(self.n_c*self.b_extra)
        np.random.seed(rng_seed)
        theta_noise = np.cumsum(np.random.normal(0, np.sqrt(2 * self.Dr * self.dt), (self.n_t, self.n_C)), axis=0)
        noise_cells = np.dstack((np.cos(theta_noise), np.sin(theta_noise)))
        noise = np.zeros((self.n_t, n_c_extra, 2))
        noise[:,:self.n_C]= noise_cells
        self.noise = noise



    def remove_repeats(self, tri, n_c):
        """
        For a given triangulation (nv x 3), remove repeated entries (i.e. rows)

        The triangulation is first re-ordered, such that the first cell id 
        referenced is the smallest. Achieved via the function order_tris. 
        (This preserves the internal order -- i.e. CCW)

        Then remove repeated rows via lexsort.

        NB: order of vertices changes via the conventions of lexsort

        Inspired by...
        https://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array

        :param tri: (nv x 3) matrix, the triangulation
        :return: triangulation minus the repeated entries (nv* x 3) (where nv* is the new # vertices).
        """
        tri = order_tris(np.mod(tri,n_c))
        sorted_tri = tri[np.lexsort(tri.T), :]
        row_mask = np.append([True], np.any(np.diff(sorted_tri, axis=0), 1))

        return sorted_tri[row_mask]



    def reset_k2s(self):
        self.k2s = get_k2_boundary(self.tris, self.v_neighbours).ravel()
        self.v_neighbours_flat = self.v_neighbours.ravel()
        self.b_neighbour_mask = (self.k2s > 0) * (self.v_neighbours_flat > 0)


    def triangulate(self,x,recalc_angles=False):
        self.Angles = tri_angles(x, self.tris)

        if type(self.k2s) is list:
            self._triangulate(x)
            if recalc_angles is True:
                self.Angles = tri_angles(x, self.tris)
        elif not ((self.Angles[self.v_neighbours_flat, self.k2s] + self.Angles.ravel()) < np.pi)[self.b_neighbour_mask].all():
            self._triangulate(x)
            if recalc_angles is True:
                self.Angles = tri_angles(x, self.tris)
        else:
            self.Cents = x[self.tris]
            self.vs = circumcenter(self.Cents)
            self.neighbours = self.vs[self.v_neighbours]



    def _triangulate(self,x):
        """
        Calculates the triangulation on the set of points x.

        Stores:
            self.n_v = number of vertices (int32)
            self.tris = triangulation of the vertices (nv x 3) matrix.
                Cells are stored in CCW order. As a convention, the first entry has the smallest cell id
                (Which entry comes first is, in and of itself, arbitrary, but is utilised elsewhere)
            self.vs = coordinates of each vertex; (nv x 2) matrix
            self.v_neighbours = vertex ids (i.e. rows of self.vs) corresponding to the 3 neighbours of a given vertex (nv x 3).
                In CCW order, where vertex i {i=0..2} is opposite cell i in the corresponding row of self.tris
            self.neighbours = coordinates of each neighbouring vertex (nv x 3 x 2) matrix

        :param x: (nc x 2) matrix with the coordinates of each cell
        """

        T = Delaunay(x)
        tri = T.simplices
        neighbours = T.neighbors

        b_cells = np.zeros(self.n_c)
        b_cells[self.n_C:] = 1

        three_b_cell_mask = b_cells[tri].sum(axis=1)==3
        tri = tri[~three_b_cell_mask]

        neigh_map = np.cumsum(~three_b_cell_mask)-1
        neigh_map[three_b_cell_mask] = -1
        neigh_map = np.concatenate((neigh_map,[-1]))

        neighbours = neighbours[~three_b_cell_mask]
        neighbours = neigh_map[neighbours]

        #6. Store outputs
        self.tris = tri
        self.n_v = tri.shape[0]
        self.Cents = x[self.tris]
        self.vs = circumcenter(self.Cents)

        #7. Manually calculate the neighbours. See doc_string for conventions.
        self.v_neighbours = neighbours
        self.neighbours = self.vs[neighbours]
        self.neighbours[neighbours == -1] = np.nan

        self.reset_k2s()



    def _triangulate_periodic(self, x):
        """
        Calculates the periodic triangulation on the set of points x.

        Stores:
            self.n_v = number of vertices (int32)
            self.tris = triangulation of the vertices (nv x 3) matrix.
                Cells are stored in CCW order. As a convention, the first entry has the smallest cell id
                (Which entry comes first is, in and of itself, arbitrary, but is utilised elsewhere)
            self.vs = coordinates of each vertex; (nv x 2) matrix
            self.v_neighbours = vertex ids (i.e. rows of self.vs) corresponding to the 3 neighbours of a given vertex (nv x 3).
                In CCW order, where vertex i {i=0..2} is opposite cell i in the corresponding row of self.tris
            self.neighbours = coordinates of each neighbouring vertex (nv x 3 x 2) matrix

        :param x: (nc x 2) matrix with the coordinates of each cell
        """

        # 1. Tile cell positions 9-fold to perform the periodic triangulation
        #   Calculates y from x. y is (9nc x 2) matrix, where the first (nc x 2) are the "true" cell positions,
        #   and the rest are translations

        # TODO: clean-up the grid formation
        grid_x, grid_y = np.mgrid[-1:2,-1:2]
        grid_x[0,0], grid_x[1,1] = grid_x[1,1], grid_x[0,0]
        grid_y[0,0], grid_y[1,1] = grid_y[1,1], grid_y[0,0]
        grid_xy = np.array([grid_x.ravel(), grid_y.ravel()]).T

        grid_xy[:,0] *= self.domain_size[0]
        grid_xy[:,1] *= self.domain_size[1]

        y = make_y(x, grid_xy)

        # 2. Perform the triangulation on y
        #   The **triangle** package (tr) returns a dictionary, containing the triangulation.
        #   This triangulation is extracted and saved as tri
        T = Delaunay(y)
        tri = T.simplices

        # Del = Delaunay(y)
        # tri = Del.simplices
        n_c = x.shape[0]

        # 3. Find triangles with **at least one** cell within the "true" frame 
        # (i.e. with **at least one** "normal cell").
        # Ignore entries with -1, a quirk of the **triangle** package, which 
        # denotes boundary triangles.
        # Generate a mask -- one_in -- that considers such triangles.
        # Save the new triangulation by applying the mask -- new_tri.
        tri = tri[(tri != -1).all(axis=1)]
        one_in = (tri < n_c).any(axis=1)
        new_tri = tri[one_in]

        # 4. Remove repeats in new_tri
        #   new_tri contains repeats of the same cells, i.e. in cases where triangles straddle a boundary
        #   Use remove_repeats function to remove these. Repeats are flagged up as entries with the same trio of
        #   cell ids, which are transformed by the mod function to account for periodicity. See function for more details
        n_tri = self.remove_repeats(new_tri,n_c)

        # tri_same = (self.tris == n_tri).all()

        #6. Store outputs
        self.n_v = n_tri.shape[0]
        self.tris = n_tri
        self.Cents = x[self.tris]
        self.vs = circumcenter_periodic(self.Cents, self.domain_size)

        #7. Manually calculate the neighbours. See doc_string for conventions.
        n_neigh = get_neighbours(n_tri)
        self.v_neighbours = n_neigh
        self.neighbours = self.vs[n_neigh]



    def check_boundary(self, x):
        """
        For a non-periodic simulation using boundary particles, dynamically update
        the number/position of particles to preserve cell shape continuity while 
        also minimizing the number of boundary particles.

        Provided the cell aggregate is completely contiguous, then **check_boundary** 
        ensures boundary particles form a single ring (i.e. where, within the set 
        of triangles featuring two boundary cells, each boundary cell is represented 
        in two such triangles)

        Performs two steps.

        1. Add extra boundary cells. Calculate the angle that "real" cells make 
           with pairs of boundary cells (within triangulation).
           Reflect the "real" cell over the line made by the pair of boundary cells 
           if this angle > 90 degs

        2. Remove extra cells.
            Remove boundary cells that are not connected to at least one "real" cell.

        :param x: Cell centroids (n_c x 2)
        :return: Updated cell centroids (n_c x 2)
        """
        b_cells = np.zeros(self.n_c)
        b_cells[self.n_C:] = 1
        vBC = b_cells[self.tris]
        considered_triangles = vBC.sum(axis=1) == 2
        add_extra = ((self.Angles*(1-vBC)>np.pi/2).T*considered_triangles.T).T
        if add_extra.any():
            I,J = np.nonzero(add_extra)
            for k,i in enumerate(I):
                j = J[k]
                xs = x[self.tris[i]]
                re = xs[np.mod(j-1,3)] - xs[np.mod(j+1,3)]
                re = re/np.linalg.norm(re)
                re = np.array([re[1],-re[0]])
                rpe = xs[j]
                x_new = 2*np.dot(xs[np.mod(j-1,3)]-rpe,re)*re + rpe
                x = np.vstack((x,x_new))
            self.n_c = x.shape[0]
            self._triangulate(x)
            self.assign_vertices()

        C = get_C_boundary(self.n_c,self.CV_matrix)
        #
        # #Remove extra cells
        # keep_mask = C[self.n_C:, :self.n_C].sum(axis=1)>0 #I'm assuming this is the same thing. This removes all boundary centroids that are not connected to at least one real centroid.
        # if keep_mask.any():
        #     c_keep = np.nonzero(keep_mask)[0]
        #     x = np.concatenate((x[:self.n_C],x[c_keep + self.n_C]))
        #     self.n_c = x.shape[0]
        #     self._triangulate(x)
        #     self.assign_vertices()
        #

        #Remove all boundary particles not connected to exactly two other boundary particles
        remove_mask = C[self.n_C:, self.n_C:].sum(axis=1)!=2
        if remove_mask.any():
            c_keep = np.nonzero(~remove_mask)[0]
            x = np.concatenate((x[:self.n_C],x[c_keep + self.n_C]))
            self.n_c = x.shape[0]
            self._triangulate(x)
            self.assign_vertices()
            self.Angles = tri_angles(x, self.tris)
        #
        # remove_mask = C[self.n_C:, self.n_C:].sum(axis=1)==0
        # if remove_mask.any():
        #     c_keep = np.nonzero(~remove_mask)[0]
        #     x = np.concatenate((x[:self.n_C],x[c_keep + self.n_C]))
        #     self.n_c = x.shape[0]
        #     self._triangulate(x)
        #     self.assign_vertices()
        #     self.Angles = tri_angles(x, self.tris)

        return x



    def get_P(self, neighbours, vs):
        """
        Calculates perimeter of each cell

        :param neighbours: (nv x 3 x 2) matrix considering the coordinates of each neighbouring vertex of each vertex
        :param vs: (nv x 2) matrix considering coordinates of each vertex
        :return: self.P saves the perimeters of each cell
        """
        self.P = get_P(vs, neighbours, self.CV_matrix, self.n_c)
        
        return self.P



    def get_P_periodic(self, neighbours, vs):
        """
        Identical to **get_P** but accounts for periodic triangulation

        Calculates perimeter of each cell (considering periodic boundary conditions)

        :param neighbours: (nv x 3 x 2) matrix considering the coordinates of each neighbouring vertex of each vertex
        :param vs: (nv x 2) matrix considering coordinates of each vertex
        :return: self.P saves the perimeters of each cell
        """
        self.P = get_P_periodic(vs, neighbours, self.CV_matrix, self.domain_size, self.n_c)

        return self.P



    def get_A(self, neighbours, vs):
        """
        Calculates area of each cell

        :param neighbours: (nv x 3 x 2) matrix considering the coordinates of each neighbouring vertex of each vertex
        :param vs: (nv x 2) matrix considering coordinates of each vertex
        :return: self.A saves the areas of each cell
        """
        self.A = get_A(vs, neighbours, self.CV_matrix, self.n_c)

        return self.A



    def get_A_periodic(self, neighbours, vs):
        """
        Identical to **get_A** but accounts for periodic triangulation.

        Calculates area of each cell (considering periodic boundary conditions)

        :param neighbours: (nv x 3 x 2) matrix considering the coordinates of each neighbouring vertex of each vertex
        :param vs: (nv x 2) matrix considering coordinates of each vertex
        :return: self.A saves the areas of each cell
        """
        self.A = get_A_periodic(vs, neighbours, self.Cents, self.CV_matrix, self.domain_size, self.n_c)

        return self.A



    def get_F_periodic(self, neighbours, vs):
        """
        Calculate the forces acting on each cell via the SPV formalism.

        Detailed explanations of each stage are described in line, but overall the strategy leverages the chain-rule
        (vertex-wise) decomposition of the expression for forces acting on each cell. Using this, contributions from
        each vertex is calculated **on the triangulation**, without the need for explicitly calculating the voronoi
        polygons. Hugely improves efficiency

        :param neighbours: Positions of neighbouring vertices (n_v x 3 x 2)
        :param vs: Positions of vertices (n_v x 2)
        :return: F
        """
        J_CW = self.J[self.tris, roll_forward(self.tris)]
        J_CCW = self.J[self.tris, roll_reverse(self.tris)]
        A0 = np.array([self.A0[j] for j in self.c_types])
        P0 = np.array([self.P0[j] for j in self.c_types])
        F = get_F_periodic(vs, neighbours, self.tris, self.CV_matrix, self.n_v, \
                           self.n_c, self.domain_size, J_CW, J_CCW, self.A, self.P, \
                           self.Cents, self.kappa_A, self.kappa_P, A0, P0)

        return F



    def get_F(self, neighbours, vs):
        """
        Identical to **get_F_periodic** but instead accounts for boundaries and neglects periodic triangulation.

        Calculate the forces acting on each cell via the SPV formalism.

        Detailed explanations of each stage are described in line, but overall the strategy leverages the chain-rule
        (vertex-wise) decomposition of the expression for forces acting on each cell. Using this, contributions from
        each vertex is calculated **on the triangulation**, without the need for explicitly calculating the voronoi
        polygons. Hugely improves efficiency

        :param neighbours: Positions of neighbouring vertices (n_v x 3 x 2)
        :param vs: Positions of vertices (n_v x 2)
        :return: F
        """
        J = self.J_large[:self.n_c,:self.n_c]
        # J[:self.n_C,:self.n_C] = self.J
        # J[self.n_C:,self.n_C:] = 0
        J_CW = J[self.tris, roll_forward(self.tris)]
        J_CCW = J[self.tris, roll_reverse(self.tris)]
        F = get_F(vs, neighbours, self.tris, self.CV_matrix, self.n_v, self.n_c, self.L, J_CW, J_CCW, self.A, self.P, self.Cents, self.kappa_A, self.kappa_P, self.A0, self.P0,self.n_C,self.kappa_B,self.l_b0)
        return F



    def simulate(self, print_every=1000, variable_param=False, \
                 output_dir="plots", rng_seed=1):
        """
        Evolve the SPV.

        Stores:
            self.x_save = Cell centroids for each time-step (n_t x n_c x 2), where n_t is the number of time-steps
            self.tri_save = Triangulation for each time-step (n_t x n_v x 3)


        :param print_every: integer value to skip printing progress every "print_every" iterations.
        :param variable_param: Set this to True if kappa_A,kappa_P are vectors rather than single values
        :return: self.x_save
        """

        # TODO: detect whether kappa_* are vectors and act accordingly.
        if variable_param is True:
            F_get = self.get_F_periodic_param
        else:
            F_get = self.get_F_periodic
        
        # Get number of time steps.
        n_t = self.t_span.size
        self.n_t = n_t
        
        # Perform initial triangulation of the cell centers 'x'. 
        x = self.x0.copy()
        self._triangulate_periodic(x)
        self.x = x.copy()
        self.x_save = np.zeros((n_t, self.n_c, 2))
        self.tri_save = np.zeros((n_t, self.tris.shape[0], 3), dtype=np.int32)
        self.assign_vertices()      # fills self.CV_matrix
        self.get_A_periodic(self.neighbours, self.vs)   # get cell areas. TODO: better function name
        self.get_P_periodic(self.neighbours, self.vs)   # get cell perimeters. TODO: the same

        # Generate random motility noise.
        self.generate_noise(rng_seed)       # fills self.noise
        noise = np.transpose( [self.noise[0,:][:,0], self.noise[0,:][:,1]] )

        # TODO: initial state is not stored currently

        for i in range(n_t):
            if (i%print_every == 0):
                print('%.1f%% --- Cell areas mean, min., max.: %.2f, %.2f, %.2f' \
                      %(i/n_t*100, np.mean(self.A), np.min(self.A), np.max(self.A)))
            
            # Triangulate cell centers. Calling Delaunay at every step for now.
            self._triangulate_periodic(x)
            # self.triangulate_periodic(x)        # equiangulate
            self.tri_save[i] = self.tris
            self.assign_vertices()      # fills self.CV_matrix
            self.get_A_periodic(self.neighbours, self.vs)   # get cell areas. TODO: better function name
            self.get_P_periodic(self.neighbours, self.vs)   # get cell perimeters. TODO: the same

            # Compute the main motility force and a weak short-range repulsion
            # to avoid unphysical configurations:
            F = F_get(self.neighbours, self.vs)
            F_soft = weak_repulsion(self.Cents, self.a, self.k, self.CV_matrix, \
                                    self.n_c, self.domain_size)
            
            # Set per-cell noise given the multipliers for each cell type.
            v0 = [self.v0[j] for j in self.c_types]
            noise = np.transpose( [v0*self.noise[i,:][:,0], v0*self.noise[i,:][:,1]] )

            # Update cell centroid positions with explicit Euler.
            x += self.dt*(F + F_soft + noise)       # Barton eq. (13)
            # x += self.dt*(F + F_soft + self.v0*self.noise[i])     # scalar self.v0

            # Impose periodicity to cell positions (Delaunay vertices).
            # E.g., if a cell position is x = 1.2, while domain width is 
            # L = 1.0, the correct position in a periodic domain is 
            # x_new = mod(x,L) = 0.2.
            x = np.mod(x, self.domain_size)

            self.x = x
            self.x_save[i] = x

            # Plot current state every integer time point.
            if (np.mod(i*self.dt, 1) == 0):
                data = np.array([self.domain_size, self.x_save[i], self.c_types, self.tri_save[i], self.A, self.P], dtype=object)
                np.save("%s/%d.npy" %(output_dir, i), data, allow_pickle=True)
                # plot_step( self.x_save[i], i, self.domain_size, self.c_types, self.colors, 
                #            self.plot_scatter, self.tris, dir_name=output_dir )
        
        print("Simulation complete")

        return self.x_save, self.tri_save



    def simulate_boundary(self, print_every=1000, do_F_bound=True, \
                          output_dir="plots", rng_seed=1):
        """
        Evolve the SPV but using boundaries.

        Stores:
            self.x_save = Cell centroids for each time-step (n_t x n_c x 2), where n_t is the number of time-steps
            self.tri_save = Triangulation for each time-step (n_t x n_v x 3)


        :param print_every: integer value to skip printing progress every "print_every" iterations.
        :param b_extra: Set this to >1. Defines the size of x_save to account for variable numbers of (boundary) cells.
            if b_extra = 2, then x_save.shape[1] = 2*n_c (at t=0)
        :return: self.x_save
        """
        n_t = self.t_span.size
        self.n_t = n_t
        x = self.x0.copy()
        self._triangulate(x)
        self.assign_vertices()
        x = self.check_boundary(x)
        self.x = x.copy()
        self.x_save = np.ones((n_t,int(self.n_c*self.b_extra),2))*np.nan
        self.tri_save = -np.ones((n_t,int(self.tris.shape[0]*self.b_extra),3),dtype=np.int32)
        self.generate_noise_boundary()

        if do_F_bound is True:
            for i in range(n_t):
                if i % print_every == 0:
                    print(i / n_t * 100, "%")
                self.triangulate(x,recalc_angles=True)
                self.assign_vertices()
                x = self.check_boundary(x)
                self.tri_save[i,:self.tris.shape[0]] = self.tris
                self.get_A(self.neighbours,self.vs)
                self.get_P(self.neighbours,self.vs)
                F = self.get_F(self.neighbours,self.vs)
                # F_bend = get_F_bend(self.n_c, self.CV_matrix, self.n_C, x, self.zeta)
                F_soft = weak_repulsion_boundary(self.Cents,self.a,self.k, self.CV_matrix,self.n_c,self.n_C)
                F_bound = boundary_tension(self.Gamma_bound,self.n_C,self.n_c,self.Cents,self.CV_matrix)
                
                # Set per-cell noise given the multipliers for each cell type.
                v0 = [self.v0[j] for j in self.c_types]
                # NOTE: last element of v0 for boundary particles
                vp = np.repeat(1, self.n_c-self.n_C) * self.v0[-1]
                v0 = np.concatenate((v0, vp))   # motility for cells + particles

                noise = np.transpose( [v0*self.noise[i,:x.shape[0]][:,0], \
                                       v0*self.noise[i,:x.shape[0]][:,1]] )

                x += self.dt*(F + F_soft + noise + F_bound)
                # x += self.dt*(F + F_soft + self.v0*self.noise[i,:x.shape[0]] + F_bound)
                                # + F_bend + F_bound
                
                self.x = x
                self.x_save[i,:x.shape[0]] = x

                # Plot current state every integer time point.
                if (np.mod(i*self.dt, 1) == 0):
                    plot_step( self.x_save[i], i, self.domain_size, self.c_types, self.colors, 
                               self.plot_scatter, self.tris, dir_name=output_dir, an_type="boundary" )
        
        else:
            for i in range(n_t):
                if i % print_every == 0:
                    print(i / n_t * 100, "%")
                self.triangulate(x, recalc_angles=True)
                self.assign_vertices()
                x = self.check_boundary(x)
                self.tri_save[i, :self.tris.shape[0]] = self.tris
                self.get_A(self.neighbours, self.vs)
                self.get_P(self.neighbours, self.vs)
                F = self.get_F(self.neighbours, self.vs)
                F_soft = weak_repulsion_boundary(self.Cents, self.a, self.k, self.CV_matrix, self.n_c, self.n_C)

                # Set per-cell noise given the multipliers for each cell type.
                v0 = [self.v0[j] for j in self.c_types]
                # NOTE: last element of v0 for boundary particles
                vp = np.repeat(1, self.n_c-self.n_C) * self.v0[-1]
                v0 = np.concatenate((v0, vp))   # motility for cells + particles

                noise = np.transpose( [v0*self.noise[i,:x.shape[0]][:,0], \
                                       v0*self.noise[i,:x.shape[0]][:,1]] )

                x += self.dt*(F + F_soft + noise + F_bound)
                # x += self.dt * (F + F_soft + self.v0*self.noise[i,:x.shape[0]])

                self.x = x
                self.x_save[i, :x.shape[0]] = x

                # Plot current state every integer time point.
                if (np.mod(i*self.dt, 1) == 0):
                    plot_step( self.x_save[i], i, self.domain_size, self.c_types, self.colors, 
                               self.plot_scatter, self.tris, dir_name=output_dir, an_type="boundary" )
        
        print("Simulation complete")
        
        return self.x_save
