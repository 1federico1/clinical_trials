from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
import numpy as np
import random

def lp_sensitivity(matrix, p=2):
    if not isinstance(matrix, np.ndarray):
        return np.max(np.linalg.norm(matrix.todense(), p, axis=1))
    else:
        return np.max(np.linalg.norm(matrix, p, axis=1))


def compute_noise_matrix(eps, delta, projection_matrix, y):
    # noise matrix should be the same dimensions of projected matrix y
    w2 = lp_sensitivity(projection_matrix)
    tmp = (np.sqrt(2 * np.log(1 / (2 * delta)) + eps)) / (eps)
    sigma = w2 * tmp
    return np.random.normal(0, sigma, y.shape), sigma


def private_projection(data, eps, delta, k, is_sparse=True):

    if is_sparse:
        transformer = SparseRandomProjection(n_components=k, random_state=int(random.random() * 100), density=1. / 3.)
    else:
        transformer = GaussianRandomProjection(n_components=k, random_state=int(random.random() * 100))
    transformed_data = transformer.fit_transform(data)
    projection_matrix = transformer.components_

    # construct random nxk noise matrix based on privacy parameters eps, delta and projection matrix P
    # in order to satisfy dp the entries of the noise matrix should be drawn from N(0, sigma^2), with
    # sigma >= w2(P) * (sqrt(2(ln(1/2delta)) + eps) / eps
    # assume delta < 1/2

    noise_matrix, sigma = compute_noise_matrix(eps, delta, projection_matrix, transformed_data)

    return transformed_data + noise_matrix, projection_matrix, sigma


