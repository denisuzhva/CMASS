# adj_mat_gen.py


import numpy as np


def example_squares_16():
    """
    Example adjacency matrix corresponding to a graph with 16 nodes connected into 4 cyclic squares, 
    which are themselves inter-connected in a larger suqare with its corners being the four mentioned cyclic agent clusters.

    Args:
        adj_mat_empty: Empty adjacency matrix

    Returns:
        adj_mat: Ready adjacency matrix
    """
    n = 16
    adj_mat = np.zeros((n, n))

    # 1st cluster
    adj_mat[0, 3] = 1.
    adj_mat[1, 0] = 1.
    adj_mat[2, 1] = 1.
    adj_mat[3, 2] = 1.
    adj_mat[3, 4] = 1.

    # 2nd cluster
    adj_mat[0+4, 3+4] = 1.
    adj_mat[1+4, 0+4] = 1.
    adj_mat[2+4, 1+4] = 1.
    adj_mat[3+4, 2+4] = 1.
    adj_mat[6, 11] = 1.

    # 3rd cluster
    adj_mat[0+8, 3+8] = 1.
    adj_mat[1+8, 0+8] = 1.
    adj_mat[2+8, 1+8] = 1.
    adj_mat[3+8, 2+8] = 1.
    adj_mat[9, 14] = 1.

    # 4th cluster
    adj_mat[0+12, 3+12] = 1.
    adj_mat[1+12, 0+12] = 1.
    adj_mat[2+12, 1+12] = 1.
    adj_mat[3+12, 2+12] = 1.
    adj_mat[12, 1] = 1.

    return adj_mat