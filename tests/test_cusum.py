import numpy as np

from change_point_algorithms.online_detection.cusum import cusum_alg_v0_rust_hybrid

def generate_normal_points(mean: float, stddev: float, num_points: int):
    """
    :param mean:
    :param stddev:
    :param num_points:
    :return:
    """
    rng = np.random.default_rng()
    return rng.normal(mean, stddev, num_points)

class TestRustCusum:
    safe_mean = 0.0
    safe_std_dev = 1.0
    threshold = 5.0
    alpha = 0.95
    unsafe_mean = 50.0
    unsafe_std_dev = 1.0 # 2.0
    num_unknowns = 1_000

    def test_cusum_v0_rust_hybrid_all_normal(self):
        my_unknowns = generate_normal_points(self.safe_mean, self.safe_std_dev * 0.01, self.num_unknowns)
        model_gen = cusum_alg_v0_rust_hybrid(
            my_unknowns, self.safe_mean, self.safe_std_dev, self.threshold, self.alpha)
        predictions = [item for item in model_gen]
        # print(predictions)
        assert all([item == False for item in
                    predictions]), f'Model predicted that {[item for item in predictions].count(True)} were change points.'

    def test_cusum_v0_rust_hybrid_all_abnormal(self):
        my_unknowns = generate_normal_points(
            self.unsafe_mean, self.unsafe_std_dev, self.num_unknowns)
        model_gen = cusum_alg_v0_rust_hybrid(
            my_unknowns, self.safe_mean, self.safe_std_dev, self.threshold, self.alpha)
        predictions = [item for item in model_gen]
        # print(predictions)
        # Skip first 10 to give cusum time to detect.
        assert all([item == True for item in
                    predictions[10:]]), f'Model predicted that {[item for item in predictions].count(False)} were change points.'
        # assert all([item == True for item in
        #             predictions]), f'Model predicted that {[item for item in predictions].count(False)} were change points.'
