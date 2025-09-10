import numpy as np

from change_point_algorithms.online_detection.bocpd import bocpd_rust_hybrid


def get_parameters():
    alpha, beta, mu, kappa, lamb = 1.0, 1.0, 0.0, 2.0, 2.0
    # std_dev = 1.0
    # points = rng.normal(mu, std_dev, size=vec_size)
    return alpha, beta, mu, kappa, lamb

def generate_normal_points(mean: float, stddev: float, num_points: int):
    """
    :param mean:
    :param stddev:
    :param num_points:
    :return:
    """
    rng = np.random.default_rng()
    return rng.normal(mean, stddev, num_points)


class TestRustModels:
    alpha, beta, mu, kappa, lamb = get_parameters()
    num_unknowns = 100
    safe_mean = mu
    safe_stddev = 1.0
    # safe_stddev: float = (beta * (kappa + 1.0)) / (kappa * alpha)
    unsafe_mean = 100.0
    unsafe_stddev = 1.0


    def test_bocpd_rust_hybrid_all_normal(self):
        my_unknowns = generate_normal_points(
            self.safe_mean, self.safe_stddev, self.num_unknowns)
        model_gen = bocpd_rust_hybrid(
            my_unknowns, self.mu, self.kappa, self.alpha, self.beta, self.lamb)
        predictions = [item for item in model_gen]
        # print(predictions)
        assert all([item == False for item in
                    predictions]), f'Model predicted that {[item for item in predictions].count(True)} were change points.'

    def test_bocpd_rust_hybrid_all_abnormal(self):
        my_unknowns = generate_normal_points(
            self.unsafe_mean, self.unsafe_stddev, self.num_unknowns)
        model_gen = bocpd_rust_hybrid(
            my_unknowns, self.mu, self.kappa, self.alpha, self.beta, self.lamb)
        predictions = [item for item in model_gen]
        # print(predictions)
        assert all([item == True for item in
                    predictions]), f'Model predicted that {[item for item in predictions].count(False)} were change points.'