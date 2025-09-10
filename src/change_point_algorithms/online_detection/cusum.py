# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:24:29 2023

@author: localuser
"""
from enum import Enum

from change_point_algorithms import _change_point_algorithms
from change_point_algorithms.online_detection.model_helpers import (
    detection_to_intervals_for_generator_v1_with_progress, detection_to_intervals_for_generator_v1)


class CusumAlgVersion(Enum):
    ALG_V0 = 'v0'
    ALG_V1 = 'v1'

def cusum(
        time, data, mean, sigma, alpha, beta, shock_intervals=None,
        non_shock_intervals=None):
    """ """
    # Instantiating variables
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    shock = False
    begin = 0
    # next_data = list()
    m_bar = [mean]
    h = 5 * sigma
    cp, cn = [0], [0]
    var = sigma**2
    accumulator = data[0]
    for idx, val in enumerate(data[1:], start=1):
        accumulator += val
        m_bar.append(beta * m_bar[idx - 1] - (1 - beta) * val)
        m_bar[0], m_bar[1] = mean, mean
        mean_p = accumulator / (idx + 1)
        diff = m_bar[idx - 1] - mean_p
        alpha_diff = alpha * diff
        alpha_diff_var, alpha_diff_half = alpha_diff / var, alpha_diff * 0.5
        cp.append(max(0, cp[idx - 1] + alpha_diff_var * (val - diff - alpha_diff_half)))
        cn.append(max(0, cn[idx - 1] - alpha_diff_var * (val + diff + alpha_diff_half)))
        # check if change occurred
        attack_likely = (cp[idx] > h or cn[idx] > h)
        if attack_likely:
            cp[idx] = 0
            cn[idx] = 0
        if attack_likely and not shock:
            non_shocks.append((time[begin], time[idx]))
            begin = idx
            shock = True
        elif not attack_likely and shock:
            shocks.append((time[begin], time[idx]))
            begin = idx
            shock = False
    # Check if remaining segment is shock or not
    if shock:
        shocks.append((time[begin], time[-1]))
    else:
        non_shocks.append((time[begin], time[-1]))
    return shocks, non_shocks


def simple_cusum(
        time, data, mean, std_dev, k=1, h=5, shock_intervals=None,
        non_shock_intervals=None):
    """ """
    # Instantiating variables
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    cp, cn = [0], [0]
    values = (data - mean) / std_dev
    shock = False
    begin = 0
    for idx, value in enumerate(values[1:], start=1):
        cp.append(max(0, value - k + cp[idx - 1]))
        cn.append(max(0, -value - k + cn[idx - 1]))
        attack_likely = (cn[idx] > h or cp[idx] > h)
        if attack_likely and not shock:
            non_shocks.append((time[begin], time[idx - 1]))
            begin = idx
            shock = True
        elif not attack_likely and shock:
            shocks.append((time[begin], time[idx - 1]))
            begin = idx
            shock = False
    # Check if remaining segment is shock or not
    if shock:
        shocks.append((time[begin], time[-1]))
    else:
        non_shocks.append((time[begin], time[-1]))
    return shocks, non_shocks


def cusum_alg(
        time, data, mean, std_dev, h, alpha, shock_intervals=None,
        non_shock_intervals=None):
    """ """
    raise NotImplementedError()


def cusum_alg_generator(data, mean, std_dev, h, alpha):
    """ """
    cp, cn, d, mu = [0], [0], [0], [data[0]]
    # First point is always True
    yield True
    h_val = h * std_dev
    variance = std_dev**2
    d = 0
    scalar, weight_no_diff = 1 + alpha * 0.5, alpha / variance
    for idx, event in enumerate(data[1:], start=1):
        weight = d * weight_no_diff
        cp.append(max(0, cp[-1] + weight * (event - d * scalar)))
        cn.append(min(0, cn[-1] - weight * (event + d * scalar)))
        d = mu[-1] - mean  # mean_p
        mu.append((1 - alpha) * mu[idx - 1] + alpha * event)
        # check if likely
        attack_likely = (cp[idx] > h_val or cn[idx] < -h_val)
        if attack_likely:
            cp[idx], cn[idx] = 0, 0
        yield attack_likely

def cusum_alg_v0_rust_hybrid(data, mean: float, std_dev: float, h: float, alpha: float):
    """ """
    prob_threshold = h * std_dev
    model = _change_point_algorithms.CusumV0(mean, std_dev**2, alpha, h)
    for idx, event in enumerate(data):
        model.update(event)
        probability = model.predict(event)
        is_attack = probability > prob_threshold
        yield is_attack

def cusum_alg_v1(
        time, data, mean, std_dev, h, alpha):
    """ Return an array of model predictions for each unknown."""
    model_gen = cusum_alg_v1_generator(data,mean, std_dev, h, alpha)
    out = [item for item in model_gen]
    return out


def cusum_alg_v1_generator(data, mean, std_dev, h, alpha):
    """ """
    cp, cn, mu = [0], [0], [data[0]]
    # First point is always True
    yield True
    h_val = h * std_dev
    variance = std_dev ** 2
    for idx, event in enumerate(data[1:], start=1):
        prev_mu = mu[-1]
        dev_shift = (prev_mu - mean) / variance
        mean_mean = (alpha * prev_mu + mean) * 0.5
        target = prev_mu + mean_mean
        mu.append(alpha * prev_mu - (1 - alpha) * event)
        cp.append(max(0, cp[-1] + dev_shift * (event - target)))
        cn.append(min(0, cn[-1] - dev_shift * (event + target)))
        # Check if attack occurred
        attack_likely = (cp[idx] >= h_val or cn[idx] <= -h_val)
        if attack_likely:
            cp[idx], cn[idx] = 0, 0
        yield attack_likely

def cusum_alg_v1_rust_hybrid(data, mean, std_dev, h, alpha):
    """ """
    prob_threshold = h * std_dev
    model = _change_point_algorithms.CusumV1(mean, std_dev, alpha, h)
    for idx, event in enumerate(data):
        model.update(event)
        probability = model.predict(event)
        is_attack = probability >= prob_threshold
        yield is_attack

def get_cusum_from_generator(time, data, mean, std_dev, h, alpha, version=None, with_progress=False):
    """ Return output of cusum algorithm using generator."""
    begin = 0
    default_alg = CusumAlgVersion.ALG_V0
    if version is None:
        alg_version = default_alg
    else:
        try:
            alg_version = CusumAlgVersion(version)
        except ValueError:
            print(f'Algorithm version {version} not a valid option for cusum. Defaulting to {default_alg}')
            alg_version = default_alg
    match alg_version:
        case CusumAlgVersion.ALG_V0:
            model_gen = cusum_alg_generator(data, mean, std_dev, h, alpha)
        case CusumAlgVersion.ALG_V1:
            model_gen = cusum_alg_v1_generator(data, mean, std_dev, h, alpha)
        case _:
            raise ValueError(f'Invalid cusum algorithm version: {version}')
    if with_progress:
        shocks, non_shocks = detection_to_intervals_for_generator_v1_with_progress(
            time, begin, model_gen, len(data))
    else:
        shocks, non_shocks = detection_to_intervals_for_generator_v1(
            time, begin, model_gen)
    return shocks, non_shocks

