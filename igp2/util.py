import numpy as np


def get_curvature(points: np.ndarray):
    """
    Gets the curvature of a 2D path
    based on https://en.wikipedia.org/wiki/Curvature#In_terms_of_a_general_parametrization

    Args:
        points: nx2 array of points

    Returns: curvature
    """
    gamma = np.array(points)
    s = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(gamma, axis=0), axis=1))))
    ds = np.gradient(s).reshape((-1, 1))
    d_gamma_ds = np.gradient(gamma, axis=0) / ds
    d_2_gamma_ds_2 = np.gradient(d_gamma_ds, axis=0) / ds
    kappa = np.linalg.det(np.dstack([d_gamma_ds, d_2_gamma_ds_2])) / np.linalg.norm(d_gamma_ds, axis=1) ** 3
    return kappa
