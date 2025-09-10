# from .change_point_algorithms import *
#
# __doc__ = change_point_algorithms.__doc__
# if hasattr(change_point_algorithms, "__all__"):
#     __all__ = change_point_algorithms.__all__

from change_point_algorithms import _change_point_algorithms
__doc__ = _change_point_algorithms.__doc__
if hasattr(_change_point_algorithms, "__all__"):
    __all__ = _change_point_algorithms.__all__

from change_point_algorithms._change_point_algorithms import (
    BocpdModel, EmModel, EmLikelihoodCheck, CusumV0, CusumV1,
    build_em_model, build_em_early_stop_model
)
