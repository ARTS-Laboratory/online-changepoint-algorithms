import math
import warnings
from collections import deque
from collections.abc import Iterable

import numpy as np
from numba import njit, vectorize
from change_point_algorithms import _change_point_algorithms

from change_point_algorithms.online_detection.model_helpers import (
    detection_to_intervals_for_generator_v1,
    detection_to_intervals_for_generator_v1_with_progress)


def bocpd_generator(data: np.typing.ArrayLike | Iterable[float], mu: float, kappa: float, alpha: float, beta: float, lamb: float):
    """ Generator for Bayesian Online Change Point Detection Algorithm."""
    my_data: np.ndarray = np.asarray(data)
    maxes = deque((0,), maxlen=2)
    run_length_arr = np.array([0], dtype=np.uint32)
    probabilities = np.array([1.0])
    alpha_arr, beta_arr, mu_arr, kappa_arr = np.array([alpha]), np.array([beta]), np.array([mu]), np.array([kappa])
    for idx, event in enumerate(my_data):
        probabilities, alpha_arr, beta_arr, mu_arr, kappa_arr, run_length_arr = calculate_probabilities(
            event, alpha_arr, beta_arr, mu_arr, kappa_arr, run_length_arr, probabilities, lamb, trunc_threshold=1e-32)
        max_idx = find_max_cp(probabilities)
        maxes.append(run_length_arr[max_idx])
        if maxes[-1] < maxes[0]:
            # reset params
            probabilities = np.asarray([1.0])
            run_length_arr = np.asarray([0], dtype=np.uint32)
            # maxes = [0]
            alpha_arr, beta_arr, mu_arr, kappa_arr = (
                np.array([alpha]), np.array([beta]), np.array([mu]), np.array([kappa]))
        else:
            # update
            alpha_arr, beta_arr, mu_arr, kappa_arr = update_no_attack_arr(
                event, alpha_arr, beta_arr, mu_arr, kappa_arr, alpha, beta, mu, kappa)
        # Calculate probability of change point
        attack_probs = calculate_prior_arr(event, alpha_arr, beta_arr, mu_arr, kappa_arr)
        val_prob = np.dot(attack_probs, probabilities)
        is_attack = val_prob <= 0.05
        yield is_attack

def bocpd_rust_hybrid(
        data: np.typing.ArrayLike, mu: float, kappa: float, alpha: float,
        beta: float, lamb: float, threshold=1e-8, with_cache=True):
    """ Generator for Bayesian Online Change Point Detection Algorithm using Rust class."""
    prob_threshold = 0.05
    my_data = np.asarray(data)
    model = _change_point_algorithms.BocpdModel(alpha, beta, mu, kappa, with_cache, threshold)
    for idx, event in enumerate(my_data):
        model.update(event, lamb)
        probability: float = model.predict(event)
        is_attack = probability <= prob_threshold
        yield is_attack

def calculate_probabilities(
        event, alpha, beta, mu, kappa, run_lengths, probabilities, lamb,
        trunc_threshold=1e-16):
    """ """
    hazard = hazard_function(lamb)
    priors = np.empty_like(alpha)
    calculate_prior_arr_inplace(event, alpha, beta, mu, kappa, priors)
    new_probabilities = np.zeros(probabilities.size + 1)
    # here we define the type as uint32, this is arbitrary and might need to be changed later
    new_run_lengths = np.zeros(run_lengths.size + 1, dtype=np.uint32)
    # Multiply probabilities by their priors
    priors *= probabilities
    new_probabilities[1:] += priors
    # should be fine to multiply entire vector if first element is zero
    new_probabilities *= (1 - hazard)
    new_probabilities[0] += priors.sum()
    new_probabilities[0] *= hazard
    # Normalize probabilities
    if (prob_sum := new_probabilities.sum()) != 0.0:
        new_probabilities /= prob_sum
    # Match the run length values with the probabilities
    # new_run_lengths[0] = 0  # don't need this line since array initialized to zeros
    new_run_lengths[1:] += run_lengths
    new_run_lengths[1:] += 1
    # Truncate near zero values
    # trunc = new_probabilities < trunc_threshold
    # new_probabilities[trunc] = 0.0
    threshold_filter = new_probabilities > trunc_threshold
    threshold_filter[0] = True
    new_probabilities = new_probabilities[threshold_filter]
    new_run_lengths = new_run_lengths[threshold_filter]
    threshold_filter = threshold_filter[1:]
    new_alpha, new_beta, new_mu, new_kappa = alpha[threshold_filter], beta[threshold_filter], mu[threshold_filter], kappa[threshold_filter]
    # new_alpha, new_beta, new_mu, new_kappa = alpha, beta, mu, kappa
    return new_probabilities, new_alpha, new_beta, new_mu, new_kappa, new_run_lengths


@njit
def update_no_attack_arr(
        event: float, alpha_arr: np.ndarray, beta_arr: np.ndarray,
        mu_arr: np.ndarray, kappa_arr: np.ndarray, alpha: float, beta: float,
        mu: float, kappa: float):
    """ """
    size = alpha_arr.size + 1
    # update
    mu_p = np.empty(shape=size)
    kappa_p = np.empty(shape=size)
    alpha_p = np.empty(shape=size)
    beta_p = np.empty(shape=size)

    kappa_p[1:] = kappa_arr + 1
    alpha_p[1:] = alpha_arr + 0.5
    kappa_plus = kappa_arr + 1
    mu_p[1:] = (kappa_arr * mu_arr + event) / kappa_plus
    beta_p[1:] = beta_arr + kappa_arr * np.square(event - mu_arr) / (2 * kappa_plus)
    mu_p[0] = mu
    kappa_p[0] = kappa
    alpha_p[0] = alpha
    beta_p[0] = beta
    return alpha_p, beta_p, mu_p, kappa_p


@njit
def calculate_prior_arr(point, alphas, betas, mus, kappas):
    """ Return student's T distribution PDF given parameters of inverse gamma distribution."""
    return t_func_arr(point, mus, ((betas * (kappas + 1.0)) / (alphas * kappas)), 2 * alphas)


@njit
def calculate_prior_arr_v1(point, alphas, betas, mus, kappas):
    """ Return student's T distribution PDF for given parameters of inverse gamma distribution."""
    t_values = calculate_prior_helper(point, alphas, betas, mus, kappas)
    t_values /= beta_numba(0.5, alphas)
    return t_values


@njit
def calculate_prior_helper(point, alphas, betas, mus, kappas):
    """ """
    denom = 2 * betas * (kappas + 1.0) / kappas
    t_values = (point - mus)**2 / denom
    t_values += 1.0
    exponent = -(alphas + 0.5)
    t_values **= exponent
    t_values /= np.sqrt(denom)
    return t_values


# @profile
@njit
def calculate_prior_arr_inplace(point, alphas, betas, mus, kappas, out):
    """ """
    calculate_prior_helper_inplace(point, alphas, betas, mus, kappas, out)
    out /= beta_numba(0.5, alphas)


@njit
def calculate_prior_helper_inplace(point, alphas, betas, mus, kappas, out):
    """ """
    arr = np.empty((2, alphas.size))
    denom = arr[0]
    exponent = arr[1]
    denom[:] = 2 * betas * (kappas + 1.0) / kappas
    out[:] = (point - mus)**2 / denom
    out += 1.0
    exponent[:] = -(alphas + 0.5)
    out **= exponent
    out /= np.sqrt(denom)


def find_max_cp(probs):
    return np.argmax(probs)


# @njit
def hazard_function(lam: float):
    return 1 / lam


@njit
def t_func_arr(x_bar, mu_arr, s_arr, n_arr):
    """ """
    s_n_arr = s_arr * n_arr
    n_half = n_arr * 0.5
    t_values = ((x_bar - mu_arr)**2 / s_n_arr + 1.0) ** (-(n_half + 0.5))

    t_values /= (np.sqrt(s_n_arr) * beta_numba(0.5, n_arr / 2))
    return t_values


@vectorize(['float64(float64, float64)', 'float32(float32, float32)'], cache=False, nopython=True)
def beta_numba(val_1, val_2):
    """ Return vectorized function for """
    return math.exp(math.lgamma(val_1) + math.lgamma(val_2) - math.lgamma(val_1 + val_2))


def get_bocpd_from_generator(
        time, data, mu, kappa, alpha, beta, lamb, with_progress=False):
    """ Return data interval predictions for data run on Bayesian Online Change Point Detection Algorithm."""
    # Instantiating variables
    begin = 0
    # try to use rust version, if it's not in the wheel fall back to python implementation
    try:
        bocpd_model_gen = bocpd_rust_hybrid(data, mu, kappa, alpha, beta, lamb)
        if with_progress:
            shocks, non_shocks = detection_to_intervals_for_generator_v1_with_progress(
                time, begin, bocpd_model_gen, len(data))
        else:
            shocks, non_shocks = detection_to_intervals_for_generator_v1(
                time, begin, bocpd_model_gen)
    except NameError:
        warnings.warn('Exception occurred, reverting to python')
        bocpd_model_gen = bocpd_generator(
                data, mu, kappa, alpha, beta, lamb)
        if with_progress:
            shocks, non_shocks = detection_to_intervals_for_generator_v1_with_progress(
                time, begin, bocpd_model_gen, len(data))
        else:
            shocks, non_shocks = detection_to_intervals_for_generator_v1(
                time, begin, bocpd_model_gen)
    return shocks, non_shocks
