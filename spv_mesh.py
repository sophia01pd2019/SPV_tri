"""
Generic mesh initialization functions.

TODO: Turn this into a mesh representation for the SPV simulation.
"""

import numpy as np



def init_square_lattice( rows, cols, rng, noise=0.0 ):
    """
    Constructs a lattice of square elements with optional noise.

    Parameters:
        rows: number of elements in vertical
        cols: number of elements in horizontal
        rng: random generator (np.random.detault_rng)
        noise: amplitude of the noise (optional)

    Returns 
        Cell coordinates.
    """

    x = np.linspace(0, cols-1, cols)
    y = np.linspace(0, rows-1, rows)
    xv, yv = np.meshgrid(x, y)
    points = np.array([xv.flatten(), yv.flatten()]).T

    # np.random.seed(rng_seed)
    # noise = np.random.normal(0, noise, np.shape(points))
    points += rng.normal(0, noise, np.shape(points))

    return points
