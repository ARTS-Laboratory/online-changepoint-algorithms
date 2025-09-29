from collections.abc import Sequence
from typing import Optional

from change_point_algorithms import EmLikelihoodCheck

type NormalTuple = tuple[float, float, float]

def build_em_early_stop_model(normal: NormalTuple, abnormals: Sequence[NormalTuple], arr_sizes: list[int], epochs: int) -> EmLikelihoodCheck:
    """ Return an Expectation Maximization model with early stopping for parameter updates.
    :param normal: A 3-tuple of (mean, standard deviation, probability of occurrence)
    :param abnormals: List of 3-tuples (mean, standard deviation, probability of occurrence)
    :param arr_sizes: List representing the number of samples for each distribution.
     The first size is for the normal parameter distribution. The remaining correspond to the abnormal case(s).
    :param epochs: The maximum number of iterations to perform for each parameter update.
    :return: Expectation Maximization model with early stopping. The model update stops early when the change in likelihoods is negligible.
    """

class BocpdModel:
    """ A class implementing Bayesian Online Change Point Detection.
    """
    def __init__(self, alpha: float, beta: float, mu: float, kappa: float, with_cache: bool, threshold: Optional[float]):
        """
        :param alpha:
        :param beta:
        :param mu:
        :param kappa:
        :param with_cache:
        :param threshold:
        """

    def update(self, point: float, lamb: float):
        """
        :param point: Observation used to update model.
        :param lamb: Input to hazard function. 1 / lamb.
        :return:
        """

    def predict(self, point: float) -> float:
        """
        :param point: Latest observation.
        :return: Prediction of model. Weighted sum of likelihoods of observing given point.
        """


class CusumV0:
    """ A class that implements a version of Cumulative Summation.
    """
    def __init__(self, mean: float, variance: float, alpha: float, threshold: float):
        """
        :param mean:
        :param variance:
        :param alpha:
        :param threshold:
        """

    def update(self, point: float):
        """
        :param point: Observation used to update model.
        :return:
        """

    def predict(self, _point: float) -> float:
        """
        :param _point: Not used for prediction.
        :return: Max cumulative deviation from mean.
        """


class CusumV1:
    """ A class that implements a version of Cumulative Summation.
    """
    def __init__(self, mean: float, std_dev: float, h: float, alpha: float):
        """
        :param mean:
        :param std_dev:
        :param h:
        :param alpha:
        """

    def update(self, point: float):
        """
        :param point: Observation used to update model.
        :return:
        """

    def predict(self, _point: float) -> float:
        """
        :param _point: Not used for prediction.
        :return: Max cumulative deviation from mean.
        """
