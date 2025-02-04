from BayesianAnalysis.DataFeeder import DataFeeder
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import pymc_bart as pmb
import arviz as az
from BayesianAnalysis.Conf import TARGET_ACC, N_TUNE, N_CORE, N_SAMPLES, N_CHAINS, BAYESIAN_RESULTS_MODEL_PRE_PATH, \
    SAMPLES_VI, XS_VALUES, BAYESIAN_RESULTS_PDP_AFTER_PATH, BAYESIAN_RESULTS_PPC_PATH, BAYESIAN_RESULTS_VI_PATH, \
    BAYESIAN_RESULTS_MODEL_AFTER_PATH, BAYESIAN_RESULTS_MODEL_POST_PATH, BAYESIAN_RESULTS_PDP_POST_PATH, BAYESIAN_RESULTS_PDP_PRE_PATH

from Utils.VisualizatioStyle import myStyle

myStyle()

RANDOM_SEED = 1945
np.random.seed(RANDOM_SEED)
import cloudpickle as cpkl

results_path = "F:\\users\\prasetia\\data\\Children\\children_sync\\data\\"

time = "pre"
segment = "discussion"
n_model = 200


if time=="after":
    BAYESIAN_RESULTS = BAYESIAN_RESULTS_MODEL_AFTER_PATH
    PDP_RESULTS = BAYESIAN_RESULTS_PDP_AFTER_PATH
elif time=="post":
    BAYESIAN_RESULTS = BAYESIAN_RESULTS_MODEL_POST_PATH
    PDP_RESULTS = BAYESIAN_RESULTS_PDP_POST_PATH
else:
    BAYESIAN_RESULTS = BAYESIAN_RESULTS_MODEL_PRE_PATH
    PDP_RESULTS = BAYESIAN_RESULTS_PDP_PRE_PATH


if __name__ == '__main__':
    label = str(n_model) + "m_" + segment

    # check convergence
    # with open(BAYESIAN_RESULTS + "\\" + time + "_features_BART_" + label + ".pkl", 'rb') as handle:
    #     idata = cpkl.load(handle)
    #
    # pmb.plot_convergence(idata, var_name="mu")
    plt.show()
    with open(BAYESIAN_RESULTS + "\\" + time + "_features_X_" + label + ".pkl", 'rb') as handle:
        X = cpkl.load(handle)

    with open(BAYESIAN_RESULTS + "\\" + time + "_features_y_" + label + ".pkl", 'rb') as handle:
        y = cpkl.load(handle)

    with open(BAYESIAN_RESULTS + "\\" + time + "_features_mu_" + label + ".pkl", 'rb') as handle:
        rng = np.random.default_rng(RANDOM_SEED)
        all_trees = cpkl.load(handle)
        mu = pmb.utils._sample_posterior(all_trees, X=X.values, rng=rng, size=50, shape=1)

    pmb.plot_pdp(all_trees, X=X, Y=y, grid=(4, 3), random_seed=RANDOM_SEED, var_discrete=[0, 6, 7, 8],
                 xs_values=XS_VALUES, path=PDP_RESULTS, label=label)
    plt.show()
