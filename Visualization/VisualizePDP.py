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
from Utils.Conf import DATA_PATH
from Utils.VisualizatioStyle import myStyle

myStyle()

RANDOM_SEED = 1945
np.random.seed(RANDOM_SEED)
import cloudpickle as cpkl
import os

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
    with open(BAYESIAN_RESULTS + "\\" + time + "_features_BART_" + label + ".pkl", 'rb') as handle:
        idata = cpkl.load(handle)

    pmb.plot_convergence(idata, var_name="mu")
    plt.savefig(os.path.join(PDP_RESULTS, time+"_"+segment+".pdf"),  format='pdf')
    plt.close()
    with open(os.path.join(BAYESIAN_RESULTS, time + "_features_X_" + label + ".pkl"), 'rb') as handle:
        X = cpkl.load(handle)


    with open(os.path.join(BAYESIAN_RESULTS, time + "_features_mu_" + label + ".pkl"), 'rb') as handle:
        rng = np.random.default_rng(RANDOM_SEED)
        all_trees = cpkl.load(handle)
        # mu = pmb.utils._sample_posterior(all_trees, X=X.values, rng=rng, size=50, shape=1)
    print("Start plotting")
    #pmb.plot_ice(all_trees, centered=False, X=X, Y=y, grid=(3, 3), random_seed=RANDOM_SEED, var_discrete=[0, 5, 6], color_mean="#DC267F", color="#785EF0",  alpha=0.1, path=PDP_RESULTS, label=label)
    # # plt.show()
    # pre
    pmb.plot_pdp(all_trees, X=X,  grid=(3, 3), random_seed=RANDOM_SEED, var_discrete=[0, 5, 6],
                 xs_values=XS_VALUES, path=PDP_RESULTS, label=label, xs_interval="insample")

    # post
    # pmb.plot_pdp(all_trees, X=X,  grid=(3, 3), random_seed=RANDOM_SEED,
    #              xs_values=XS_VALUES, path=PDP_RESULTS, label=label, xs_interval="insample")

    # not used
    # pmb.plot_pdp(all_trees, X=X,  grid=(3, 3), random_seed=RANDOM_SEED,
    #              xs_values=XS_VALUES, path=PDP_RESULTS, label=label)

    del all_trees
    # pmb.plot_ice(all_trees, X=X, Y=y, random_seed=RANDOM_SEED,
    #               path=PDP_RESULTS, label=label)
    # plt.show()
