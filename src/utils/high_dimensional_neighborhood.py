import os
import pickle
from time import time

import numpy as np
from openTSNE import affinity
from openTSNE.nearest_neighbors import Annoy
from scipy.sparse import csr_matrix


def distance_matrix_and_annoy(data, num_neighbors, epsilon=False, num_threads=1, rnd_state=42):
    """
    TODO Docstring for distance_matrix_and_annoy
    """
    # Create an Annoy object to get the neighborhood indices and the corresponding distances
    annoy = Annoy(
        data=data,
        k=num_neighbors,
        metric="euclidean",
        n_jobs=num_threads,
        random_state=rnd_state,
        verbose=False
    )
    neighbors, distances = annoy.build()
    # Remove explicit zero entries
    if epsilon:
        distances[distances == 0.0] = epsilon
    # Convert the information to a CSR matrix
    row_indices = np.repeat(
        np.arange(data.shape[0]),
        num_neighbors
    )
    data_distance_matrix = csr_matrix(
        (distances.flatten(), (row_indices, neighbors.flatten())),
        shape=(data.shape[0], data.shape[0])
    )
    return data_distance_matrix, annoy


def high_dimensional_distance_matrix_and_affinities(
        data_to_be_embedded, num_neighbors, perplexity, epsilon=False, num_threads=1, rnd_state=42
):
    """
    TODO Docstring for high_dimensional_distance_matrix_and_affinities.
    """
    high_dimensional_distance_matrix, annoy = distance_matrix_and_annoy(
        data_to_be_embedded, num_neighbors, epsilon, num_threads, rnd_state
    )
    # Compute the affinities (P matrix) to be used in t-SNE
    affinities = affinity.PerplexityBasedNN(
        knn_index=annoy,
        perplexity=perplexity,
        metric="euclidean",
        n_jobs=num_threads,
        random_state=rnd_state,
        verbose=False
    )
    return high_dimensional_distance_matrix, affinities


def load_or_compute_high_dimensional_distance_matrix_and_affinities(
        data_directory, data_name, data_to_be_embedded, num_neighbors, perplexity, epsilon=False, num_threads=1,
        rnd_state=42
):
    """
    TODO Docstring for load_load_or_compute_high_dimensional_distance_matrix_and_affinities.
    """
    # Determine the high-dimensional affinities (P matrix) of the data
    if (os.path.isfile(data_directory + data_name + f"_perplexity_{perplexity}_affinities.pkl")
            and os.path.isfile(
                data_directory + data_name + f"_perplexity_{perplexity}_high_dimensional_distance_matrix.pkl")):
        # Found the high-dimensional distance matrix and the affinities from a previous run, load those
        print("Loading high-dimensional distance matrix and affinities from saved pickle.")
        with (open(data_directory + data_name + f"_perplexity_{perplexity}_high_dimensional_distance_matrix.pkl", "rb")
              as high_dimensional_distance_matrix_file):
            high_dimensional_distance_matrix = pickle.load(high_dimensional_distance_matrix_file)
        with open(data_directory + data_name + f"_perplexity_{perplexity}_affinities.pkl", "rb") as affinity_file:
            affinities = pickle.load(affinity_file)
    else:
        # Could not find high-dimensional distance matrix or the affinities, compute and store them
        print("Computing affinities from scratch.")
        affinities_start = time()
        high_dimensional_distance_matrix, affinities = high_dimensional_distance_matrix_and_affinities(
            data_to_be_embedded, num_neighbors, perplexity, epsilon, num_threads, rnd_state
        )
        with (open(data_directory + data_name + f"_perplexity_{perplexity}_high_dimensional_distance_matrix.pkl", "wb")
              as high_dimensional_distance_matrix_file):
            pickle.dump(high_dimensional_distance_matrix, high_dimensional_distance_matrix_file)
        with open(data_directory + data_name + f"_perplexity_{perplexity}_affinities.pkl", "wb") as affinity_file:
            pickle.dump(affinities, affinity_file)
        print(f"Computing affinities took {time() - affinities_start} seconds.")

    return high_dimensional_distance_matrix, affinities
