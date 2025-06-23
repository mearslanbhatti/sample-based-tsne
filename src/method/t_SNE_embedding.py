"""
Wrapper methods to compute t-SNE embeddings, both as a base version for comparison and the multigrid version.
"""

from time import time

import numpy as np
from openTSNE import TSNEEmbedding, affinity


def base_tsne(
        initial_embedding, affinities, negative_gradient_method="fft", num_iterations_early_exaggeration=250,
        num_iterations_optimization=750, n_jobs=1, rnd_state=42, callbacks=None, callbacks_every_iters=50, verbose=False
):
    """
    Perform a basic t-SNE embedding on the data. That is, first perform a number of "early exaggeration" iterations,
    followed by a number of non-exaggerated iterations.
    TODO complete docstring for parameters

    :param ndarray initial_embedding:
    :param str negative_gradient_method: One of "fft" or "bh", specifying the usage of the corresponding acceleration.
    :param int callbacks:
    :param int callbacks_every_iters:
    :param bool verbose:
    :return: The time it took to compute the embedding and a 2D embedding of the data.
    """
    start = time()

    # Set up the optimization
    t_sne_embedding = TSNEEmbedding(
        initial_embedding,
        affinities,
        negative_gradient_method=negative_gradient_method,
        n_jobs=n_jobs,
        callbacks=callbacks,
        callbacks_every_iters=callbacks_every_iters,
        verbose=verbose,
        random_state=rnd_state
    )

    # Initial optimization with exaggeration
    embedding_after_early_exaggeration = t_sne_embedding.optimize(
        n_iter=num_iterations_early_exaggeration,
        exaggeration=12,
        momentum=0.5
    )

    # Final optimization without exaggeration
    final_embedding = embedding_after_early_exaggeration.optimize(
        n_iter=num_iterations_optimization,
        exaggeration=1,
        momentum=0.5
    )

    end = time()

    return end - start, final_embedding


def mg_tsne(
        initial_embedding, affinities, sample_rate, data_to_be_embedded, sample_perplexity,
        sample_ids=None, sample_affinities=None, sample_embedding=None,
        num_iterations_early_exaggeration_sample=250, num_iterations_optimization_sample=750,
        num_iterations_after_prolongation=100, n_jobs=1, rnd_state=42, return_sampling_and_all_embeddings=False,
        callbacks=None, callbacks_every_iters=50, verbose=False
):
    """
    TODO Add docstring for this method.
    """
    start = time()

    # Sample the data
    start_sample = time()
    if sample_ids is None:
        sample_ids = np.random.choice(
            np.arange(0, initial_embedding.shape[0]),
            size=int(initial_embedding.shape[0] * sample_rate),
            replace=False
        )
        sample_ids.sort()
    sampled_initial_embedding = initial_embedding[sample_ids, :]
    sampled_data = data_to_be_embedded[sample_ids, :]
    
    # compute affinities for the sampled data
    if sample_affinities is None:
        sample_affinities = affinity.PerplexityBasedNN(
            sampled_data,
            perplexity=sample_perplexity,
            metric="euclidean",
            n_jobs=n_jobs,
            random_state=rnd_state,
            verbose=verbose
        )

    if sample_embedding is None:
        # Set up the optimization
        sample_embedding = TSNEEmbedding(
            sampled_initial_embedding,
            sample_affinities,
            negative_gradient_method="fft",
            n_jobs=n_jobs,
            callbacks=callbacks,
            callbacks_every_iters=callbacks_every_iters,
            verbose=verbose,
            random_state=rnd_state
        )

        # Initial optimization with exaggeration
        sample_embedding = sample_embedding.optimize(
            n_iter=num_iterations_early_exaggeration_sample,
            exaggeration=12,
            momentum=0.5
        )

        # Further optimization without exaggeration
        sample_embedding = sample_embedding.optimize(
            n_iter=num_iterations_optimization_sample,
            exaggeration=1,
            momentum=0.5
        )

    print(f"Sampling and sample embedding took {time() - start_sample} seconds.")

    # Prolongation of the embedded sample to full size
    start_prolongation = time()
    prolongated_embedding = np.empty(initial_embedding.shape)
    prolongated_embedding[:] = np.NAN
    prolongated_embedding[sample_ids, :] = sample_embedding
    for i in range(prolongated_embedding.shape[0]):  # update points' position
        if np.isnan(prolongated_embedding[i, 0]):  # only if the point was not part of the coarse sample
            point_affinities = affinities.P[i, :].toarray().flatten()  # Get the affinity values
            sample_affinities = point_affinities[sample_ids]  # Only use those that are in the sampling
            # Put the point at the embedding position of its nearest high-dimensional neighbor that was sampled
            prolongated_embedding[i, :] = prolongated_embedding[sample_ids[np.argpartition(sample_affinities, -1)[-1]]]

    print(f"Prolongation took {time() - start_prolongation} seconds.")

    # Set up final optimization
    start_final_embedding = time()
    full_y = TSNEEmbedding(
        prolongated_embedding,
        affinities,
        negative_gradient_method="fft",
        n_jobs=n_jobs,
        callbacks=callbacks,
        callbacks_every_iters=callbacks_every_iters,
        verbose=verbose,
        random_state=rnd_state
    )

    # Final optimization without exaggeration
    full_y = full_y.optimize(
        n_iter=num_iterations_after_prolongation,
        exaggeration=1,
        momentum=0.5
    )

    print(
        f"Final embedding took {time() - start_final_embedding} seconds and finished with kld={full_y.kl_divergence}."
    )

    end = time()

    if return_sampling_and_all_embeddings:
        return end - start, sample_ids, sampled_initial_embedding, sample_embedding, prolongated_embedding, full_y
    else:
        return end - start, full_y
