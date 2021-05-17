# reconstructor.py


import numpy as np
import scipy.optimize as so
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import warnings
warnings.filterwarnings('ignore') # Disable warnings, which appear during L1 reconstruction


def get_reconst_recording(cs_recording, meas_matrix, n, threshold_factor=5):
    """
    Get reconstructed frame given a compressed one.

    Args:
        cs_recording: Compressed frame (sparse vector)
        meas_matrix: Measurement matrix
        n: Target (original) dimensionality

    Returns:
        reconst_recording: Reconstructed frame
    """
    reconst_recording  = so.linprog(c=np.ones(n), A_eq=meas_matrix, b_eq=cs_recording)['x']

    # Thresholding
    thresholds = threshold_factor * np.mean(reconst_recording)
    reconst_recording[reconst_recording < thresholds] = 0
    reconst_recording[reconst_recording > 0] = 1

    return reconst_recording


def get_cluster_data(x, delta, min_clust):
    """
    Get cluster data given a sparse vector consisting of either 0 or 1.
    The function identifies clusters according to euclidean distances between 1's.

    Args:
        x: Sparse data vector
        delta: Threshold
        min_clust: Min size of a cluster, anything below would be considered as an outlier

    Returns:
        l: Linkage data for a dendogram
        centroids: Positions of cluster centroids 
    """
    # Get positions (indices) of 1's 
    x_pos = np.nonzero(x) 
    l = linkage(np.transpose(x_pos), "single")
    x_pos = x_pos[0]
    clusters = fcluster(l, delta, criterion='distance')-1
    n_clust = clusters.max()+1
    cluster_idx = []

    # Remove small outlier clusters
    for i in range(n_clust): 
        cluster_size = np.count_nonzero(clusters == i)
        if cluster_size < min_clust:
            x_pos = np.delete(x_pos, np.where(clusters == i))
            clusters = np.delete(clusters, np.where(clusters == i))
        else:
            cluster_idx.append(i)

    # Get centroids as means of index values
    if len(clusters) == 0:
        centroids = []
    else:
        n_clust = clusters.max()
        centroids = np.zeros((len(cluster_idx)))
        for idx, i in enumerate(cluster_idx): 
            centroids[idx] = np.mean(x_pos[clusters == i])
    
    return centroids, l


def clusterization_accuracy(pred_n_clusters, true_n_clusters):
    """
    Get clusterization accuracy in a series of measurements. 

    Args:
        pred_n_clusters: Number of clusters identified by an algorithm
        true_n_clusters: True number of clusters

    Returns:
        accuracy: Calculated accuracy
    """
    total = len(pred_n_clusters)
    positive = np.sum(np.array(pred_n_clusters) == true_n_clusters)
    accuracy = positive / total

    return accuracy