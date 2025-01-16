

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection



def _voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.max(vor.points) - np.min(vor.points) # vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def plot_vor(x, ax, L, c_types, colors, plot_scatter, line_width, tri=None):
    """
    Plot the Voronoi diagram with optional Delaunay triangulation.

    :param x: Cell locations (nc x 2)
    :param ax: Matplotlib axis
    :param L: Domain size (width, height)
    :param c_types: Cell types/categories
    :param colors: Colors for each cell type
    :param plot_scatter: Boolean to enable/disable plotting cell centers
    :param line_width: Line width for Voronoi edges and Delaunay triangles
    :param tri: Delaunay triangulation (n_v x 3) array, or None to skip plotting
    """
    grid_x, grid_y = np.mgrid[-1:2, -1:2]
    y = np.vstack([x + np.array([i * L[0], j * L[1]]) for i, j in zip(grid_x.ravel(), grid_y.ravel())])

    c_types_print = np.tile(c_types, 9)
    bleed = 0.1
    valid_mask = (y < L * (1 + bleed)).all(axis=1) & (y > -L * bleed).all(axis=1)
    c_types_print = c_types_print[valid_mask]
    y = y[valid_mask]

    regions, vertices = _voronoi_finite_polygons_2d(Voronoi(y))
    ax.set(aspect=1, xlim=(0, L[0]), ylim=(0, L[1]))

    if plot_scatter:
        for j, i in enumerate(np.unique(c_types)):
            ax.scatter(
                x[c_types == i, 0],
                x[c_types == i, 1],
                color=colors[i],
                zorder=1000,
                s=10,
                edgecolor="black"
            )

    patches = []
    for i, region in enumerate(regions):
        patches.append(Polygon(vertices[region], closed=True,
                                facecolor=colors[c_types_print[i]],
                                edgecolor=(1, 1, 1, 1), linewidth=line_width))
    p = PatchCollection(patches, match_original=True)
    ax.add_collection(p)

    if tri is not None:
        for triangle in tri:
            valid_edges = []
            for j in range(3):
                a, b = triangle[j], triangle[(j + 1) % 3]

                # Identify edge endpoints
                x_a, x_b = x[a], x[b]

                # Check if edge crosses the periodic boundary
                dx = np.abs(x_a[0] - x_b[0])
                dy = np.abs(x_a[1] - x_b[1])

                if dx > L[0] * 0.5 or dy > L[1] * 0.5:
                    # Edge crosses the periodic boundary, skip it
                    continue

                # Add edge to valid edges
                valid_edges.append((a, b))

            # Plot valid edges
            for a, b in valid_edges:
                X = np.stack((x[a], x[b])).T
                ax.plot(X[0], X[1], color="black", linewidth=line_width)


def plot_vor_boundary(x, ax, L, c_types, colors, plot_scatter, line_width, tri=False):
    """
    Plot the Voronoi.

    Takes in a set of cell locs (x), tiles these 9-fold, plots the full voronoi, then crops to the field-of-view

    :param x: Cell locations (nc x 2)
    :param ax: matplotlib axis
    :param tri: Is either a (n_v x 3) np.ndarray of dtype **np.int64** defining the triangulation.
        Or **False** where the triangulation is not plotted
    """

    n_C = np.size(c_types)      # number of actual cells (= excl. boundary particles)
    x = x[~np.isnan(x[:,0])]

    c_types_print = np.ones(x.shape[0],dtype=np.int32)*-1
    c_types_print[:n_C] = c_types
    regions, vertices = _voronoi_finite_polygons_2d(Voronoi(x))

    ax.set(aspect=1, xlim=(0,L[0]), ylim=(0,L[1]))
    if type(c_types) is list:
        # ax.scatter(x[:, 0], x[:, 1],color="grey",zorder=1000)
        for region in regions:
            polygon = vertices[region]
            plt.fill(*zip(*polygon), alpha=0.4, color="grey")

    else:
        patches = []
        if plot_scatter is True:
            ax.scatter(x[:n_C, 0], x[:n_C, 1], color="black", zorder=1000)
            ax.scatter(x[n_C:, 0], x[n_C:, 1], color="grey", zorder=1000)

        for i, region in enumerate(regions):
            patches.append( Polygon(vertices[region], True, \
                                    facecolor=colors[c_types_print[i]], \
                                    edgecolor="white", alpha=0.5, linewidth=line_width) )

        p = PatchCollection(patches, match_original=True)
        # p.set_array(c_types_print)
        ax.add_collection(p)
    
    if tri is not False:
        for TRI in tri:
            for j in range(3):
                a, b = TRI[j], TRI[np.mod(j + 1, 3)]
                if (a >= 0) and (b >= 0):
                    X = np.stack((x[a], x[b])).T
                    ax.plot(X[0], X[1], color="black")



def check_forces(data, iter, F, dir_name="plots"):
    """
    Plot the forces (quiver) on each cell (voronoi)

    To be used as a quick check.

    :param x: Cell coordinates (nc x 2)
    :param F: Forces on each cell (nc x 2)
    """
    x = data
    Vor = Voronoi(x)
    fig, ax = plt.subplots()
    ax.set(aspect=1)
    voronoi_plot_2d(Vor, ax=ax)
    # ax.scatter(x[:, 0], x[:, 1])
    ax.quiver(x[:, 0], x[:, 1], F[:, 0], F[:, 1])
    fig.savefig("%s/%d.png" %(dir_name, iter), bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close(fig)


def plot_step(data, iter, domain_size, c_types, colors, plot_scatter, tri_save=None,
              dir_name="plots", an_type="periodic", line_width=1.0):
    """
    Writes the given simulation stage into a .png file with optional triangulation.

    :param data: Simulation stage.
    :param iter: Current iteration.
    :param domain_size: Domain size (width, height)
    :param c_types: Cell types/categories
    :param colors: Colors for each cell type
    :param plot_scatter: Boolean to enable/disable plotting cell centers
    :param tri_save: Delaunay triangulation to plot, or None to skip
    :param dir_name: Directory name to save simulation within
    :param an_type: Animation type -- either "periodic" or "boundary"
    :param line_width: Line width for Voronoi edges and Delaunay triangles
    """
    if an_type == "periodic":
        plot_fn = plot_vor
    elif an_type == "boundary":
        plot_fn = plot_vor_boundary
    else:
        raise ValueError("Invalid animation type. Choose 'periodic' or 'boundary'.")

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')

    plot_fn(data, ax, domain_size, c_types, colors, plot_scatter, line_width, tri=tri_save)

    output_file = f"{dir_name}/{iter}.png"
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)