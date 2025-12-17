import pymc as pm
from BayesianAnalysis.Conf import BAYESIAN_RESULTS_MODEL_POST_PATH, BAYESIAN_RESULTS_COMPARISON_PATH
import pickle
import arviz as az
import matplotlib.pyplot as plt
import cloudpickle as cpkl
import numpy as np
import os
model_list = {}
az.style.use("arviz-darkgrid")

# compare filling and remove nan

with open(BAYESIAN_RESULTS_MODEL_POST_PATH + "\\" + "post_features_BART_200m_re3_group_climate_all_02.pkl",
          'rb') as handle:
    idata_02 = cpkl.load(handle)
    print(idata_02.observed_data["observed"].size)
    print(az.loo(idata_02))
with open(BAYESIAN_RESULTS_MODEL_POST_PATH + "\\" + "post_features_BART_200m_re3_group_climate_all_05.pkl",
          'rb') as handle:
    idata_05 = cpkl.load(handle)
    print(idata_05.observed_data["observed"].size)
    # print(az.loo(idata_05))
with open(BAYESIAN_RESULTS_MODEL_POST_PATH + "\\" + "post_features_BART_200m_re3_group_climate_all_10.pkl",
          'rb') as handle:
    idata_10 = cpkl.load(handle)
    print(idata_10.observed_data["observed"].size)
    # print(az.loo(idata_10))
with open(BAYESIAN_RESULTS_MODEL_POST_PATH + "\\" + "post_features_BART_200m_re3_group_climate_all_20.pkl",
          'rb') as handle:
    idata_20 = cpkl.load(handle)
    print(idata_20.observed_data["observed"].size)
    # print(az.loo(idata_20))



