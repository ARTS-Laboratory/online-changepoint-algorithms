from collections.abc import Iterable

from change_point_algorithms import CusumV0
from change_point_algorithms.online_detection.cusum import cusum_alg_generator, cusum_alg_v0_rust_hybrid
from examples.generate_data import generate_normal

def cusum_for_sequence(arr, mean: float, std_dev: float, h: float, alpha: float):
    """ Run detection on array using Cumulative Summation algorithm."""
    model_generator = cusum_alg_generator(arr, mean, std_dev, h, alpha)
    for decision in model_generator:
        print(f"Observation is {'normal' if (not decision) else 'abnormal'}")

def rusty_cusum_for_sequence(arr, mean: float, std_dev: float, h: float, alpha: float):
    """ Run detection on array using Cumulative Summation algorithm."""
    model_generator = cusum_alg_v0_rust_hybrid(arr, mean, std_dev, h, alpha)
    for decision in model_generator:
        print(f"Observation is {'normal' if (not decision) else 'abnormal'}")

def make_use_cusum_model(arr: Iterable, mean: float, std_dev: float, alpha: float, threshold=3):
    """ Make CUSUM model and run on array."""
    decision_cutoff = threshold * std_dev
    model = CusumV0(mean, std_dev**2, alpha, threshold)
    for observation in arr:
        model.update(observation)
        prediction = model.predict(observation)
        abnormal = prediction > decision_cutoff
        print(f"Observation is {'normal' if (not abnormal) else 'abnormal'}")

def _main():
    mean = 0.0
    std_dev = 1.0
    alpha = 0.5
    threshold = 3
    num = 50
    my_arr = generate_normal(mean, std_dev, num)
    my_arr[30:41] += 10
    print("Python CUSUM algorithm")
    cusum_for_sequence(my_arr, mean, std_dev, threshold, alpha)
    print("PyO3 CUSUM algorithm")
    rusty_cusum_for_sequence(my_arr, mean, std_dev, threshold, alpha)

if __name__ == "__main__":
    _main()
