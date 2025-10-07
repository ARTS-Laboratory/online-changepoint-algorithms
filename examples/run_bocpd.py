from change_point_algorithms import BocpdModel
from change_point_algorithms.online_detection.bocpd import bocpd_rust_hybrid, bocpd_generator
from examples.generate_data import generate_normal


def bocpd_for_sequence(arr, alpha, beta, mu, kappa, lamb):
    """ Run detection on array using Bayesian Online Change Point detection."""
    model_generator = bocpd_generator(arr, mu, kappa, alpha, beta, lamb)
    for decision in model_generator:
        print(f"Observation is {'normal' if (not decision) else 'abnormal'}")

def rusty_bocpd_for_sequence(arr, alpha, beta, mu, kappa, lamb):
    """ Run detection on array using Bayesian Online Change Point detection."""
    model_generator = bocpd_rust_hybrid(arr, mu, kappa, alpha, beta, lamb)
    for decision in model_generator:
        print(f"Observation is {'normal' if (not decision) else 'abnormal'}")

def make_use_bocpd_model(arr, alpha, beta, mu, kappa, lamb, decision_cutoff=0.05):
    """ Make BOCPD model and run on array."""
    # Make the Bayesian Online Change Point Detection Algorithm
    model = BocpdModel(alpha, beta, mu, kappa, with_cache=True, threshold=1e-10)
    for observation in arr:
        model.update(observation, lamb)
        prediction = model.predict(observation)
        abnormal = prediction < decision_cutoff
        print(f"Observation is {'normal' if (not abnormal) else 'abnormal'}")
    return


def _main():
    mean = 0.0
    std_dev = 2.0
    num = 50
    my_arr = generate_normal(mean, std_dev, num)
    my_arr[30:41] += 10
    print("Python BOCPD algorithm")
    bocpd_for_sequence(my_arr, 0.5, 2.0, 0.0, 1.0, 4.0)
    print("PyO3 BOCPD algorithm")
    rusty_bocpd_for_sequence(my_arr, 0.5, 2.0, 0.0, 1.0, 4.0)


if __name__ == "__main__":
    _main()
