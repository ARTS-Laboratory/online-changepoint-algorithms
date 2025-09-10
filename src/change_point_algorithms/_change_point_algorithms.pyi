from typing import Optional

from change_point_algorithms import EmLikelihoodCheck

type NormalTuple = (float, float, float)

def build_em_early_stop_model(normal: NormalTuple, abnormals: [NormalTuple], arr_sizes: list[int], epochs: int) -> EmLikelihoodCheck:
    """ 
    :param normal: 
    :param abnormals: 
    :param arr_sizes: 
    :param epochs: 
    :return: 
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

    def predict(self, point: float):
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
