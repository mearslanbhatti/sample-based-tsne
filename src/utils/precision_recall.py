import numpy as np
from openTSNE.nearest_neighbors import Annoy
from scipy.sparse import csr_matrix

from src.utils.high_dimensional_neighborhood import distance_matrix_and_annoy


def nearest_neighbor_preservation(
        D_X, D_Y, k_start=1, k_max=30, to_return="aggregated"
):
    """
    Compute the nearest neighbor preservation of the embedding `Y` of the high-dimensional data `X`. See Section 6 of
    Pezzotti, Nicola, et al. "Hierarchical stochastic neighbor embedding." Computer Graphics Forum, Vol. 35, No. 3,
    2016. for a detailed description of the approach.
    --------
    X: ndarray
        Coordinates in original -high dimensional- space.
    Y: ndarray
        Coordinates in embedded -lower dimensional- space.
    D_X: ndarray
        Precomputed distance matrix for the high-dimensional points.
    D_Y: ndarray
        Precomputed distance matrix for the embedding.
    k_start: int
        Neighborhood size from which the TPs will be computed. Should hold k_start < k_max
    k_max: int
        Size of neighborhoods to consider.
    exact_nn: ndarray
        Whether to use exact or approximate nearest neighbors for building the
        distance matrices. Approximate methods might introduce noise in the
        precision/recall curves.
    consider_order: bool
        if True, the ordered neighborhoods will be compared for obtaining the TPs.
        If False, the neighborhoods will be compared using set intersection (order
        agnostic).
    strict: bool
        If True, TPs will be computed based on k neighborhood in the high dimensional
        space. If False, the k_max neighborhood will be used.
    to_return: str
        Which values should be returned. Either `aggregate` or `full`.

    Returns
    -------
    thresholds: list
        Thresholds used for computing precisions and recalls.
    precisions: list
        Precisions for every threshold averaged over all the points.
        For a single point, precision = TP/threshold
    recalls: list
        Recalls for every threshold averaged over all the points.
        For a single point, recall = TP/k_max
    nums_true_positives: list of lists
        Number of true positives for every threshold value and every point.
    """
    num_points = D_X.shape[0]
    num_available_nn_hd = D_X[0, :].nnz
    num_available_nn_ld = D_Y[0, :].nnz

    # Check for potential problems
    size_smaller_neighborhood = min(num_available_nn_ld, num_available_nn_hd)
    if size_smaller_neighborhood < k_max:
        print("[nearest_neighbor_preservation] Warning: k_max is {} but the size of the available neighborhoods is {}."
              "Adjusting k_max to the available neighbors.".format(k_max, size_smaller_neighborhood))
        k_max = size_smaller_neighborhood
    if k_start > k_max:
        raise Exception("[nearest_neighbor_preservation] Error: k_start is larger than k_max. Please adjust"
                        " this value to satisfy k_start <= k_max.")
    if k_start <= 0:
        print("[nearest_neighbor_preservation] Warning: k_start must be a value above 0. Setting it to 1.")
        k_start = 1

    # Compute precision recall curves for every value of k_emb
    precisions = []
    recalls = []

    # Computation of ordered neighbourhoods
    nz_D_X = D_X.nonzero()
    nz_rows_D_X = nz_D_X[0].reshape(-1, num_available_nn_hd)  # row coordinate of nz elements from D_X
    nz_cols_D_X = nz_D_X[1].reshape(-1, num_available_nn_hd)  # col coordinate of nz elements from D_X
    nz_dists_D_X = np.asarray(D_X[nz_rows_D_X, nz_cols_D_X].todense())
    sorting_ids_nz_dists_D_X = np.argsort(nz_dists_D_X, axis=1)
    sorted_nz_cols_D_X = nz_cols_D_X[nz_rows_D_X, sorting_ids_nz_dists_D_X]  # sorted cols of nz_D_X
    sorted_nz_cols_D_X = sorted_nz_cols_D_X[:, 0:k_max]  # only get NNs that will be used

    nz_D_Y = D_Y.nonzero()
    nz_rows_D_Y = nz_D_Y[0].reshape(-1, num_available_nn_ld)  # row coordinate of nz elements from D_Y
    nz_cols_D_Y = nz_D_Y[1].reshape(-1, num_available_nn_ld)  # col coordinate of nz elements from D_Y
    nz_dists_D_Y = np.asarray(D_Y[nz_rows_D_Y, nz_cols_D_Y].todense())
    sorting_ids_nz_dists_D_Y = np.argsort(nz_dists_D_Y, axis=1)
    sorted_nz_cols_D_Y = nz_cols_D_Y[nz_rows_D_Y, sorting_ids_nz_dists_D_Y]  # sorted cols of nz_D_Y
    sorted_nz_cols_D_Y = sorted_nz_cols_D_Y[:, 0:k_max]  # only get NNs that will be used

    # Compute metrics
    thresholds = np.arange(k_start, k_max+1)
    nums_true_positives = []
    index_hd = np.arange(0, k_max)

    for k_emb in thresholds:
        tps = []
        for point_id in range(num_points):  # point_id between 0 and N-1
            high_dim_arr = sorted_nz_cols_D_X[point_id, index_hd]
            low_dim_arr = sorted_nz_cols_D_Y[point_id, 0:k_emb]
            neighbourhood_intersection = np.array(np.intersect1d(high_dim_arr, low_dim_arr))
            tps.append(neighbourhood_intersection.size)

        tps = np.array(tps)  # tps for all points for a given k_emb

        precision = (tps/k_emb).mean()  # precision = TP / k_emb
        recall = (tps/k_max).mean()  # recall = TP / h_high

        if to_return == "full":
            nums_true_positives.append(tps)
        precisions.append(precision)
        recalls.append(recall)

    precisions = precisions
    recalls = recalls
    nums_true_positives = nums_true_positives

    if to_return == "aggregated":
        return thresholds, precisions, recalls
    elif to_return == "full":
        return thresholds, precisions, recalls, nums_true_positives
    else:
        raise "Unknown value for parameter `to_return`: " + str(to_return)


def precision_recall_curves(
        high_dimensional_distance_matrix, embedding, k_max, epsilon=False, num_threads=1, rnd_state=42
):
    """
    TODO Docstring for precision_recall_curves
    """
    low_dimensional_distances, _ = distance_matrix_and_annoy(
        embedding, k_max, epsilon=epsilon, num_threads=num_threads, rnd_state=rnd_state
    )
    _, precisions, recalls = nearest_neighbor_preservation(
        high_dimensional_distance_matrix, low_dimensional_distances, k_max=k_max
    )

    return precisions, recalls
