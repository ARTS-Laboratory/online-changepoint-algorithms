from collections.abc import Iterable

import numpy as np

from change_point_algorithms import EmLikelihoodCheck, build_em_early_stop_model
from change_point_algorithms.online_detection.expect_Max import em_rust_hybrid, expectation_maximization_generator
from examples.generate_data import generate_normal


# def em_for_sequence(
#         arr: Iterable[float], safe_data: np.ndarray, unsafe_data: np.ndarray,
#         safe_mean: float, safe_variance: float, unsafe_mean: float,
#         unsafe_variance: float, abnormal_prob: float, epochs: int):
#     """ Run detection on array using Expectation Maximization algorithm."""
#     model_generator = expectation_maximization_generator(
#         safe_data, unsafe_data, arr, safe_mean, unsafe_mean, safe_variance,
#         unsafe_variance, abnormal_prob, epochs=epochs)
#     for decision in model_generator:
#         print(f"Observation is {'normal' if (not decision) else 'abnormal'}")

def rusty_em_for_sequence(
        arr: Iterable[float], safe_mean: float, safe_variance: float, num_safe: int,
        unsafe_mean: float, unsafe_variance: float, num_unsafe: int,
        abnormal_prob: float, epochs: int):
    """ Run detection on array using Expectation Maximization algorithm."""
    model_generator = em_rust_hybrid(
        arr, safe_mean, safe_variance, num_safe, unsafe_mean,
        unsafe_variance, num_unsafe, abnormal_prob, epochs=epochs)
    for decision in model_generator:
        print(f"Observation is {'normal' if (not decision) else 'abnormal'}")

def make_use_em_model(
        arr, safe_mean, safe_std_dev, prob_safe, unsafe_mean, unsafe_std_dev,
        prob_unsafe, arr_sizes, epochs, early_stop_threshold=1e-8, decision_cutoff=0.95):
    """ Make Expectation Maximization model and run on array.

        This model uses early stopping to complete update steps faster.
    """
    normal_params = (safe_mean, safe_std_dev, prob_safe)
    abnormal_params = [(unsafe_mean, unsafe_std_dev, prob_unsafe)]
    model: EmLikelihoodCheck = build_em_early_stop_model(normal_params, abnormal_params, arr_sizes, epochs)
    for observation in arr:
        model.update_check_convergence(observation, early_stop_threshold)
        prediction = model.predict(observation)
        abnormal = prediction < decision_cutoff
        print(f"Observation is {'normal' if (not abnormal) else 'abnormal'}")

def _main():
    safe_mean = 0.0
    safe_std_dev = 1.0
    prob_safe = 0.7
    unsafe_mean = 20.0
    unsafe_std_dev = 1.0
    prob_unsafe = 0.3
    num_safe = 70
    num_unsafe = 30
    safe_arr = generate_normal(safe_mean, safe_std_dev, num_safe)
    unsafe_arr = generate_normal(unsafe_mean, unsafe_std_dev, num_unsafe)
    my_arr = np.concatenate((safe_arr, unsafe_arr))
    safe_data = generate_normal(safe_mean, safe_std_dev, num_safe)
    unsafe_data = generate_normal(unsafe_mean, unsafe_std_dev, num_unsafe)
    epochs = 100
    # print("Python EM algorithm")
    # em_for_sequence(
    #     my_arr, safe_data, unsafe_data, safe_mean, safe_std_dev**2,
    #     unsafe_mean, unsafe_std_dev**2, prob_unsafe, epochs)
    print("PyO3 EM algorithm")
    rusty_em_for_sequence(
        my_arr, safe_mean, safe_std_dev**2, num_safe, unsafe_mean,
        unsafe_std_dev**2, num_unsafe, prob_unsafe, epochs)

if __name__ == "__main__":
    _main()
