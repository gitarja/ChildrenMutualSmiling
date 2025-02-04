import pymc as pm
from BayesianAnalysis.Conf import BAYESIAN_RESULTS_MODEL_PRE_PATH, BAYESIAN_RESULTS_MODEL_AFTER_PATH, BAYESIAN_RESULTS_MODEL_POST_PATH
import arviz as az
import matplotlib.pyplot as plt
import pymc_bart as pmb
import pymc_bart.utils as pmbu
import numpy as np
import pytensor as pt

import cloudpickle as cpkl

RANDOM_SEED = 1945
n_model = 50

label = str(n_model) + "m_discussion"
with open(BAYESIAN_RESULTS_MODEL_AFTER_PATH + "\\" + "pre_features_BART_" + label + ".pkl", 'rb') as handle:
    idata = cpkl.load(handle)

summary = az.summary(idata, var_names=["mu"])
az.plot_ppc(idata, num_pp_samples=100, kind="cumulative", observed_rug=True)
plt.show()
pmb.plot_convergence(idata, var_name="mu")
plt.show()
# pmb.plot_pdp(mu, X=X, Y=y, grid=(5, 3), figsize=(6, 9))
