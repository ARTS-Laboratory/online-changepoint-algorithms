import math
from typing import Iterable

import numpy as np
from numba import njit

from change_point_algorithms import _change_point_algorithms

from change_point_algorithms.online_detection.model_helpers import (
    detection_to_intervals_for_generator_v1,
    detection_to_intervals_for_generator_v1_with_progress)


def expectation_maximization_generator(
        safe, not_safe, unknowns: Iterable[float], mean_1, mean_2, var_1,
        var_2, pi, epochs=1):
    """ Perform expectation maximization on one unknown.

        :param safe: Data that is known to be safe.
        :param not_safe: Data that is not known to be safe.
        :param unknowns: Collection of data that needs to be classified.
        :param float mean_1: Estimated mean for safe data.
        :param float mean_2: Estimated mean for unsafe data.
        :param float var_1: Estimated variance for safe data.
        :param float var_2: Estimated variance for unsafe data.
        :param float pi: Estimated probability that an attack has occurred.
        :param int epochs: Number of epochs to update parameters.
        :returns: Tuple of (attack classification, updated mean 1,
        updated mean 2, updated variance 1, updated variance 2,
         updated attack probability.
        :rtype: (bool, float, float, float, float, float)
        """
    data = np.concatenate((safe, not_safe, np.empty(1)))
    # Variable initialization
    size = len(data)
    mu1_hat, mu2_hat = mean_1, mean_2
    sig1_hat, sig2_hat = var_1, var_2
    pi_hat = pi
    last_attack_prob = np.empty_like(data)
    last_attack_prob[:] = -1e99
    out = np.empty((2, size))
    attack_prob, inverse = out[0], out[1]
    for unknown in unknowns:
        data[-1] = unknown # reassign last value to our new unknown
        # For some number of epochs, iterate over given data until convergence
        for idx in range(epochs):
            # Expectation
            posterior_probs_v2_inplace(
                data, pi_hat, mu2_hat, sig2_hat, mu1_hat, sig1_hat, out)
            # Maximization
            mu1_hat, mu2_hat, sig1_hat, sig2_hat, pi_hat = maximization(
                data, attack_prob, inverse, mu1_hat, mu2_hat,
                sig1_hat, sig2_hat, pi_hat, size)
            if close_enough(attack_prob, last_attack_prob):
                break
            last_attack_prob[:] = attack_prob
        is_attack = posterior_prob(
            unknown, pi_hat, mu2_hat, sig2_hat, mu1_hat, sig1_hat) > 0.01
        yield is_attack


@njit
def close_enough(a, b):
    return np.allclose(a, b) and np.allclose(b, a)


@njit
def maximization(
        data, attack_prob, inverse, mu1_hat, mu2_hat, sig1_hat, sig2_hat,
        pi_hat, size):
    """ Return updated parameter values for two normal distributions.
    """
    density, inverse_density = attack_prob.sum(), inverse.sum()
    # If all probabilities are zero for attack or not attack, no need to update
    if not (density == 0 or inverse_density == 0):
        new_mu1_hat, new_mu2_hat = update_means(
            attack_prob, inverse, density, inverse_density, data)
        new_sig1_hat, new_sig2_hat = update_variances(
            attack_prob, inverse, density, inverse_density, data, mu1_hat,
            mu2_hat)
        new_pi_hat = update_attack_prob(density, size)
        return new_mu1_hat, new_mu2_hat, new_sig1_hat, new_sig2_hat, new_pi_hat
    return mu1_hat, mu2_hat, sig1_hat, sig2_hat, pi_hat


@njit
def phi_single(value, mean, variance):
    """ Return the probability density function for a single value.

        :param float value: Value to get function for.
        :param float mean: Population mean.
        :param float variance: Population variance.
        :returns: PDF of value given.
        :rtype: float
    """
    if variance == 0.0:
        return 1.0 if value == mean else 0.0
    denom = math.sqrt(variance*2*np.pi)
    ex = math.exp(-0.5*(value - mean)**2/variance)
    return ex/denom


@njit
def phi_inplace(values, mean, variance, out):
    """ Return the probability density function for some values.

        :param ndarray values: Value to get function for.
        :param float mean: Population mean.
        :param float variance: Population variance.
        :returns: PDF of value given.
        :rtype: float
    """
    if variance == 0.0:
        for idx, value in enumerate(values):
            out[idx] = 1.0 if value == mean else 0.0
        return
    out[:] = values
    out -= mean
    out **= 2
    out *= -0.5 / variance
    np.exp(out, out)
    out /= math.sqrt(variance)


@njit
def posterior_prob(point, attack_prob, attack_mean, attack_var, normal_mean, normal_var):
    """ Calculate probability of latent variable for given data point."""
    # Probability of attack * Probability of point occurring if it was an attack
    # Divided by probability of point occurring
    num = attack_prob * phi_single(point, attack_mean, attack_var)
    denom = num + (1 - attack_prob) * phi_single(point, normal_mean, normal_var)
    if denom == 0.0:
        return num
    return num / denom


@njit
def posterior_probs_v2(points, attack_prob, attack_mean, attack_var, normal_mean, normal_var):
    """ Calculate probabilities of each latent variable for each data point."""
    num_1 = np.empty_like(points)
    phi_inplace(points, attack_mean, attack_var, num_1)
    # num_1 = phi(points, attack_mean, attack_var)
    num_1 *= attack_prob
    num_2 = np.empty_like(points)
    phi_inplace(points, normal_mean, normal_var, num_2)
    num_2 *= (1 - attack_prob)
    denom = num_1 + num_2
    normalize_probs(num_1, num_2, denom)
    return num_1, num_2


@njit
def normalize_probs(probs, inverse, denom):
    for idx, value in enumerate(denom):
        if value != 0.0:
            probs[idx] /= value
            inverse[idx] /= value


@njit
def posterior_probs_v2_inplace(points, attack_prob, attack_mean, attack_var, normal_mean, normal_var, out):
    """ Calculate probabilities of each latent variable for each data point."""
    num_1 = out[0]
    num_2 = out[1]
    phi_inplace(points, attack_mean, attack_var, out[0])
    num_1 *= attack_prob
    # new lines
    phi_inplace(points, normal_mean, normal_var, num_2)
    num_2 *= (1 - attack_prob)
    denom = num_1 + num_2
    normalize_probs(num_1, num_2, denom)


@njit
def update_means(probs: np.typing.ArrayLike, inverse: np.typing.ArrayLike, density: float, inverse_density: float, events: np.typing.ArrayLike | list[float]) -> (float, float):
    """ Return updated values for means.

        :param probs: List of probabilities of attack.
        :param inverse: List of probabilities of safe.
        :param float density: Probability density for mean 1.
        :param float inverse_density: Probability density for mean 1.
        :param events: List of events corresponding to probs.
        :returns: tuple of updated means (mean 1, mean 2)"""
    mean_1 = np.dot(inverse, events) / inverse_density
    mean_2 = np.dot(probs, events) / density
    return mean_1, mean_2


@njit
def variance_helper(probs: np.ndarray, events: np.ndarray, mean: float):
    """ """
    return np.dot(probs, np.square(events - mean))


@njit
def update_variances(
        probs: np.ndarray, inverse: np.ndarray, density: float,
        inverse_density: float, events: np.ndarray, mean_1: float,
        mean_2: float) -> tuple[float, float]:
    """ Return updated variances.


        :param probs: Array of probabilities of attack.
        :param inverse: Array of probabilities of safe.
        :param float density: Probability density for mean 1.
        :param float inverse_density: Probability density for mean 1.
        :param events: Array of events corresponding to probs.
        :param float mean_1: Approximate mean for safe data.
        :param float mean_2: Approximate mean for unsafe data.
        :returns: Tuple of mean for safe data and mean for unsafe data.
        :rtype: (float, float)
    """
    var_1 = variance_helper(inverse, events, mean_1) / inverse_density
    var_2 = variance_helper(probs, events, mean_2) / density
    return var_1, var_2


@njit
def update_attack_prob(density: float, size: int) -> float:
    """ Return updated attack probability.

        :param float density: Probability density of data.
        :param int size: Length of data vector.
        :returns: Updated attack probability.
        :rtype: float
    """
    return density / size

def em_rust_hybrid(data, safe_mean: float, safe_stddev: float, num_safe: int, unsafe_mean: float, unsafe_stddev: float, num_unsafe: int,
                   # , mean_1, mean_2, var_1, var_2,
                   pi: float, epochs=1, prob_threshold=0.05, early_stopping=False):
    """ Return decision of each observation in data as normal or abnormal using Expectation Maximization algorithm."""
    prob_threshold_normal = 1.0 - prob_threshold
    if early_stopping:
        early_stop_threshold = 1e-5
        model = _change_point_algorithms.build_em_early_stop_model(
            (safe_mean, safe_stddev, pi), [(unsafe_mean, unsafe_stddev, 1 - pi)],
            [num_safe, num_unsafe], epochs=epochs,)
        update_model = model.update_check_convergence
        predict_model = model.predict
        for idx, event in enumerate(data):
            model.update_check_convergence(event, early_stop_threshold)
            probability = model.predict(event)
            # update_model(event, early_stop_threshold)
            # probability = predict_model(event)
            is_attack = probability < prob_threshold_normal
            yield is_attack
    else:
        model = _change_point_algorithms.build_em_model(
            (safe_mean, safe_stddev, pi), [(unsafe_mean, unsafe_stddev, 1 - pi)],
            [num_safe, num_unsafe], epochs=epochs)
        update_model = model.update
        predict_model = model.predict
        for idx, event in enumerate(data):
            update_model(event)
            probability = predict_model(event)
            # model.update(event)
            # probability = model.predict(event)
            is_attack = probability < prob_threshold_normal
            yield is_attack


def get_em_from_generator(
        time, normal_obs, abnormal_obs, unknowns, mean_1=None, mean_2=None,
        var_1=None, var_2=None, pi=None, epochs=1, with_progress=False):
    """ Run expectation maximization algorithm and return detection regions."""
    # Instantiating variables
    begin = 0
    # get params theta = mu , mu2, sig, sig2, pi
    mean_1_p = mean_1 if mean_1 is not None else np.mean(normal_obs)
    mean_2_p = mean_2 if mean_2 is not None else np.mean(abnormal_obs)
    var_1_p = var_1 if var_1 is not None else np.var(normal_obs)
    var_2_p = var_2 if var_2 is not None else np.var(abnormal_obs)
    if pi is not None:
        pi_p = pi
    else:
        normal_size, ab_size = len(normal_obs), len(abnormal_obs)
        pi_p = ab_size / (normal_size + ab_size)
    # Begin algorithm loop
    my_normal_obs, my_abnormal_obs, my_unknowns = np.asarray(normal_obs), np.asarray(abnormal_obs), np.asarray(unknowns)
    em_model_gen = expectation_maximization_generator(
        my_normal_obs, my_abnormal_obs, my_unknowns, mean_1_p, mean_2_p,
        var_1_p, var_2_p, pi_p, epochs)
    if with_progress:
        shocks, non_shocks = detection_to_intervals_for_generator_v1_with_progress(
            time, begin, em_model_gen, len(unknowns))
    else:
        shocks, non_shocks = detection_to_intervals_for_generator_v1(
        time, begin, em_model_gen)
    return shocks, non_shocks
