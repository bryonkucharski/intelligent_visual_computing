from __future__ import division
import numpy as np


def computeShapeHistogram(mesh, y_min, y_max, number_of_bins):

    """
    complete this function to compute a histogram capturing the 
    distribution of surface point locations along the upright 
    axis (y-axis) in the given range [y_min, y_max] for a mesh. 
    The histogram should be normalized i.e., represent a 
    discrete probability distribution.
    
    input: 
    mesh structure from 'loadMesh' function
    => mesh.V contains the 3D locations of V mesh vertices (3xV matrix)
    => mesh.F contains the mesh faces (triangles). 
            Each triangle contains indices to three vertices (3xF matrix)

    output: 
    shape histogram (a column vector)
    """

    histogram = np.zeros((1, number_of_bins))

    """""""""""""""""""""""""""""""""
    ADD CODE HERE TO COMPUTE mesh.H
    """""""""""""""""""""""""""""""""

    return histogram

    



