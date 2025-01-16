"""Functions for writing various simulation data into txt files.
"""
import numpy as np



def write_velocity_stats( data, c_types, dt, output_file="stats_velocity.txt" ):
    """
    Computes minimum, mean and maximum velocities for each phase at all time steps.

    TODO: Implement handling of periodic boundaries. For now discarding all
    velocity values indicating boundary jump.

    :param data: array of cell centroid positions for each simulation time step
    :param c_types: cell type identifiers
    :param dt: time step between each simulation step
    :param output_file: output file name for writing the velocity statistics
    """
    
    fout = open(output_file, "w")
    
    fout.write("Time\tCell_type\tMean\tMin\tMax\tStd\n")
    for i in range(1, np.shape(data)[0]):
        v = (data[i,:,:] - data[i-1,:,:]) / dt
        
        # Compute threshold for skipping boundary jumps.
        thr = np.min([np.max(data[0,:,0]), np.max(data[0,:,1])]) / dt
        thr *= 0.9      # just something to give a little room
        
        for ctype in np.unique(c_types):
            ids = np.where(c_types == ctype)
            v_ctype = v[ids,:]
            v_norm = np.sqrt( np.sum(v_ctype**2, 2).ravel() )
            
            # Skip the boundary jumps.
            ids = np.where(v_norm > thr)[0]
            v_norm = np.delete(v_norm, ids)
            fout.write("%f\t%d\t%f\t%f\t%f\t%f\n" %(i*dt, ctype, np.mean(v_norm), \
                                                    np.min(v_norm), np.max(v_norm), \
                                                    np.std(v_norm)))
    
    fout.close()



def write_connection_matrix( tris, c_types, output_folder="" ):
    """
    Writes an nc x nc matrix for nc Delauany vertices (cells), where element (i,j)
    is a boolean (0/1) of connection between cells i and j as indicated by the
    triangulation.
    """

    idx = np.argsort(c_types)[::-1]
    idx_map = np.argsort(idx)

    n = np.size(idx)        # total number of cells
    C = np.zeros([n,n])     # connection matrix

    for k in range(0, np.shape(tris)[0]):
        t = tris[k]
        i,j = idx_map[t[0]], idx_map[t[1]]
        C[i,j] = 1
        i,j = idx_map[t[0]], idx_map[t[2]]
        C[i,j] = 1
        i,j = idx_map[t[1]], idx_map[t[2]]
        C[i,j] = 1

    # Output file name "connections_n1_n2_..." where n1 is the number of cells in
    # the 1st phase (highest c_type), n2 number of cells in 1st plus 2nd phase etc.
    # This information can then be used to interpret the cell types in the final matrix.
    output = output_folder + "/connections"    
    for i in np.unique(c_types)[::-1]:
        ntype = np.where(c_types[idx]==i)[0][-1] + 1
        output += "_" + str(ntype)

    output += ".txt"

    C = np.matrix(C)
    with open(output, 'w') as f:
        for line in C:
            np.savetxt(f, line, fmt='%d')
