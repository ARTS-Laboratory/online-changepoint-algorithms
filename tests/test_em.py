
import numpy as np

from change_point_algorithms.online_detection.expect_Max import em_rust_hybrid


def get_parameters():
    # mean_1, var_1 = 0.0, 1.0
    safe_mean, safe_stddev = 0.0, 1.0
    # mean_2, var_2 = 10.0, 2.0
    unsafe_mean, unsafe_stddev = 100.0, 2.0
    # return mean_1, var_1, mean_2, var_2
    return safe_mean, safe_stddev, unsafe_mean, unsafe_stddev

class TestRustModels:

    safe_mean, safe_stddev, unsafe_mean, unsafe_stddev = get_parameters()
    safe_size, unsafe_size = 70, 30
    pi = 0.3
    epochs = 100
    num_unknowns = 21_000

    def test_em_rust_hybrid_all_normal(self):
        rng = np.random.default_rng()
        my_unknowns = rng.normal(self.safe_mean, self.safe_stddev, self.num_unknowns)
        model_gen = em_rust_hybrid(
            my_unknowns, self.safe_mean, self.safe_stddev, self.safe_size,
            self.unsafe_mean, self.unsafe_stddev, self.unsafe_size, self.pi, epochs=self.epochs, prob_threshold=0.01, early_stopping=False)
        predictions = [item for item in model_gen]
        # print(predictions)
        assert all([item == False for item in predictions]), f'Model predicted that {[item for item in predictions].count(True)} were change points.'

    def test_em_rust_hybrid_all_normal_early_stopping(self):
        rng = np.random.default_rng()
        my_unknowns = rng.normal(self.safe_mean, self.safe_stddev, self.num_unknowns)
        model_gen = em_rust_hybrid(
            my_unknowns, self.safe_mean, self.safe_stddev, self.safe_size,
            self.unsafe_mean, self.unsafe_stddev, self.unsafe_size, self.pi, epochs=self.epochs, prob_threshold=0.01, early_stopping=True)
        predictions = [item for item in model_gen]
        # print(predictions)
        assert all([item == False for item in predictions]), f'Model predicted that {[item for item in predictions].count(True)} were change points.'

    def test_em_rust_hybrid_all_abnormal(self):
        rng = np.random.default_rng()
        my_unknowns = rng.normal(self.unsafe_mean, self.unsafe_stddev, self.num_unknowns)
        model_gen = em_rust_hybrid(
            my_unknowns, self.safe_mean, self.safe_stddev, self.safe_size,
            self.unsafe_mean, self.unsafe_stddev, self.unsafe_size, self.pi, epochs=self.epochs, prob_threshold=0.01,
            early_stopping=False)
        predictions = [item for item in model_gen]
        # print(predictions)
        assert all([item == True for item in
                    predictions]), f'Model predicted that {[item for item in predictions].count(False)} were change points.'


    def test_em_rust_hybrid_all_abnormal_early_stopping(self):
        rng = np.random.default_rng()
        my_unknowns = rng.normal(self.unsafe_mean, self.unsafe_stddev, self.num_unknowns)
        model_gen = em_rust_hybrid(
            my_unknowns, self.safe_mean, self.safe_stddev, self.safe_size,
            self.unsafe_mean, self.unsafe_stddev, self.unsafe_size, self.pi, epochs=self.epochs, prob_threshold=0.01,
            early_stopping=True)
        predictions = [item for item in model_gen]
        # print(predictions)
        assert all([item == True for item in
                    predictions]), f'Model predicted that {[item for item in predictions].count(False)} were change points.'


# def test_expectation_maximization_generator_all_normal():
#     mean_1, var_1, mean_2, var_2 = get_parameters()
#     safe_size, unsafe_size = 70, 30
#     pi = 0.3
#     epochs = 100
#     num_unknowns = 1_000
#     rng = np.random.default_rng()
#     safe = rng.normal(mean_1, math.sqrt(var_1), safe_size)
#     unsafe = rng.normal(mean_2, math.sqrt(var_2), unsafe_size)
#     my_unknowns = rng.normal(mean_1, math.sqrt(var_1), num_unknowns)
#     em_model_gen = expectation_maximization_generator(
#         safe, unsafe, my_unknowns, mean_1, mean_2,
#         var_1, var_2, pi, epochs)
#     assert all([item == False for item in em_model_gen])
#
# def test_expectation_maximization_generator_all_abnormal():
#     mean_1, var_1, mean_2, var_2 = get_parameters()
#     safe_size, unsafe_size = 70, 30
#     pi = 0.3
#     epochs = 100
#     num_unknowns = 1_000
#     rng = np.random.default_rng()
#     safe = rng.normal(mean_1, math.sqrt(var_1), safe_size)
#     unsafe = rng.normal(mean_2, math.sqrt(var_2), unsafe_size)
#     my_unknowns = rng.normal(mean_2, math.sqrt(var_2), num_unknowns)
#     em_model_gen = expectation_maximization_generator(
#         safe, unsafe, my_unknowns, mean_1, mean_2,
#         var_1, var_2, pi, epochs)
#     assert all([item == True for item in em_model_gen])