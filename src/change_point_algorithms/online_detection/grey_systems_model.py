import math

import numpy as np

from numba import njit

from change_point_algorithms.online_detection.model_helpers import (
    detection_to_intervals_for_generator_v1,
    detection_to_intervals_for_generator_v1_with_progress)


def grey_model_generator(data, window_size=1, c=3, c_ratio=300):
    """ Return Grey Model results for given data.

        :param np.ndarray data: Array of data. Data expected to be non-negative.
        :param int window_size: Size of window to iterate over array.
        :param float c: Constant multiplier for degree of grey incidence.
        :param float c_ratio: Constant multiplier for degree of ratio grey incidence.
        :returns:
        :rtype: None"""
    sample_arr = np.empty((3, window_size))
    window_1 = sample_arr[0]
    x_1 = sample_arr[1]
    z_1 = sample_arr[2]
    get_rolling_window_inplace(data, 0, window_size, window_1)
    accumulation_sequence_inplace(window_1, x_1)
    mean_sequence_inplace(x_1, z_1, alpha=0.5)
    s_1 = behavioral_sequence(z_1)
    # s_1_ratio = behavioral_sequence_ratio(z_1)
    # s_1_ratio = behavioral_sequence_ratio_2(z_1)
    s_1_rel_dist = behavior_relative_difference(z_1)
    out_arr = np.empty((3, window_size))
    window_2 = out_arr[0]
    x_2 = out_arr[1]
    z_2 = out_arr[2]
    for idx in range(0, len(data) - window_size):
        get_rolling_window_inplace(data, idx, window_size, window_2)
        accumulation_sequence_inplace(window_2, x_2)
        mean_sequence_inplace(x_2, z_2, alpha=0.5)
        s_2 = behavioral_sequence(z_2)
        # s_2_ratio = behavioral_sequence_ratio(z_2)
        # s_2_ratio = behavioral_sequence_ratio_2(z_2)
        s_2_rel_dist = behavior_relative_difference(z_2)
        degree = grey_incidence_degree(s_1, s_2, c=c)
        # degree_ratio = grey_incidence_degree_ratio(s_1_ratio, s_2_ratio, c=c_ratio)
        rel_degree = grey_incidence_degree(s_1_rel_dist, s_2_rel_dist, c=c_ratio)
        # attack_likely = degree <= 0.5 or degree_ratio <= 0.5
        attack_likely = degree <= 0.5 or rel_degree <= 0.5
        # attack_likely = min(degree, rel_degree) <= 0.5
        yield attack_likely


def grey_model_generator_2(data, window_size, w_factor, threshold):
    alp = 0.5  ##whitenization amount
    ws = 3  ##window size-compate vectors of 4 values (so similar to EM)
    threshold = 0.15 # 0.05  ## related to sensitivity-threshold
    c = data[0]  ## constant to avoid 0 division in relative grey indices
    neg_alp = 1 - alp
    for event in data[1:]:
        X = list(range(0, ws + 1))
        Z = list(range(0, ws + 1))
        ##this loop is for whitenized values of vector values then compare two consecutive vectors of window size=4
        for jdx in range(1, ws + 1):
            X[jdx] = event + X[jdx - 1]
            Z[jdx] = alp * X[jdx] + neg_alp * X[jdx - 1]
        s = 0
        sp = 0
        for k in range(1, ws):  ####calculate grey indices with L1 loss
            s = s + abs(Z[k] - Z[k - 1])
            sp = sp + abs((Z[k] + c) / (Z[0] + c) - (Z[k - 1] + c) / (Z[0] + c))

        s = s + 0.5 * abs(Z[ws] - Z[ws - 1])
        sp = sp + 0.5 * abs((Z[ws] + c) / (Z[0] + c) - (Z[ws - 1] + c) / (Z[0] + c))

        # change_likely = (1 / (1 + threshold * s)) < threshold
        # change_likely = (0.5 * (1 / (1 + threshold * sp)) + 0.5 * (1 / (1 + threshold * s))) < threshold
        change_likely = (1 / (1 + threshold * sp)) < threshold or (1 / (1 + threshold * s)) < threshold
        yield change_likely

@njit
def accumulation_sequence(window: np.ndarray):
    """ Return the accumulation over the window."""
    output = window.cumsum()
    output[-1] = window.sum()
    return output


@njit
def accumulation_sequence_inplace(window: np.ndarray, out: np.ndarray):
    """ Return the accumulation over the window."""
    # return np.add.accumulate(window)
    out[:] = window.cumsum()
    out[-1] = window.sum()


@njit
def mean_sequence(window, alpha=0.5):
    """ Return the running average over a window."""
    transformed = np.empty_like(window)
    mean_sequence_inplace(window, transformed, alpha=alpha)
    return transformed


@njit
def mean_sequence_inplace(window, out, alpha=0.5):
    """ Return the running average over a window."""
    out[0] = window[0]
    neg_alpha = 1 - alpha
    for idx in range(1, len(window)):
        out[idx] = (window[idx - 1] * alpha) + (window[idx] * neg_alpha)


@njit
def behavioral_sequence(window):
    s_0 = (window[:-1] - window[0]).sum() + 0.5 * (window[-1] - window[0])
    return s_0


@njit
def behavioral_sequence_ratio(window, offset_2=1.0):
    # This assumes window is composed only of nonnegative numbers
    # new code
    s_0 = (window[:-1] + offset_2).sum()
    s_0 += 0.5 * (window[-1] + offset_2)
    s_0 /= window[0] + offset_2
    # old code
    # s_0 = np.sum((window[:-1] + offset_2) / (window[0] + offset_2)) + 0.5 * ((window[-1] + offset_2) / (window[0] + offset_2))
    return s_0


@njit
def behavioral_sequence_ratio_2(window):
    """ """
    head = window[0]
    s_0 = 0.5 * (window[-1] - head + 1) / (0.5 * (np.abs(window[-1]) + np.abs(head) + 1))
    for item in window[:-1]:
        if item != 0.0 or window[0] != 0.0:
            s_0 += (item - head)/((np.abs(item) + np.abs(head)) * 0.5)  #
    return s_0


@njit
def behavior_relative_difference(window):
    """ """
    head = window[0]
    tail = window[-1]
    if head == 0 and tail == 0:
        s_0 = 0
    else:
        s_0 = 0.5 * abs(tail - head) / ( 0.5 * (abs(tail) + abs(head)))
    if head == 0.0:
        for item in window[1:]:
            if item != 0.0:
                s_0 += abs(item - head) / (0.5 * (abs(item) + abs(head)))
            # if head and item are zero, no difference
    else:
        for item in window[1:]:
            s_0 += abs(item - head) / (0.5 * (abs(item) + abs(head)))
    return s_0


@njit
def behavior_log_difference(window):
    """ """
    head = window[0]
    tail = window[-1]
    match (head, tail):
        case (0, 0):
            s_0 = 0.0
        case (0, tail):
            s_0 = math.inf
        case (head, 0):
            s_0 = -math.inf
        case (head, tail):
            s_0 = 0.5 * math.log(tail / head)
        case _:
            raise NotImplementedError
    for item in window[1:]:
        match (head, tail):
            case (0, 0):
                s_0 += 0.0
            case (0, tail):
                s_0 += math.inf
            case (head, 0):
                s_0 += -math.inf
            case (head, tail):
                s_0 += math.log(item / head)
            case _:
                raise NotImplementedError
    return s_0


@njit
def grey_incidence_degree(val_1, val_2, c=3.0):
    num = 1.0 + abs(val_1) + abs(val_2)
    return num / (num + c * abs(val_1 - val_2))
    # return 1 / ( 1.0 + 0.01 * abs(val_1 - val_2)/val_1)


# @njit
def grey_incidence_degree_ratio(val_1, val_2, c=1.0):
    return 1.0 / (1.0 + c * abs(val_1 - val_2))


@njit
def get_rolling_window(obs, idx, n):
    """ Get window of size n with idx being start (inclusive)."""
    # this can be either way, we look forward but can look back
    # return obs[idx - n:idx]
    return obs[idx:idx + n]


# @njit
def get_rolling_window_inplace(obs, idx, n, out):
    """ """
    out[:] = obs[idx:idx + n]


def get_grey_model(time, data, window_size=1, c=3, c_ratio=3, shock_intervals=None, non_shock_intervals=None, with_progress=False):
    """ """
    raise NotImplementedError()


def get_grey_from_generator(time, data, window_size=1, c=3, c_ratio=3, with_progress=False):
    """ Run grey systems model algorithm and return detection regions."""
    # Instantiating variables
    my_data = np.abs(data)
    begin = 0
    grey_version = 1
    match grey_version:
        case 1:
            grey_model_gen = grey_model_generator(
                my_data, window_size, c=c, c_ratio=c_ratio)
        case 2:
            grey_model_gen = grey_model_generator_2(my_data)
        case _:
            raise ValueError(f'case {grey_version} is not a valid option.')
    if with_progress:
        shocks, non_shocks = detection_to_intervals_for_generator_v1_with_progress(
            time, begin, grey_model_gen, len(data) - window_size,
            start_offset=window_size)
    else:
        shocks, non_shocks = detection_to_intervals_for_generator_v1(
            time, begin, grey_model_gen, start_offset=window_size)
    return shocks, non_shocks



