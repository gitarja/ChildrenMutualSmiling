import os.path

from BayesianAnalysis.DataFeeder import DataFeeder
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import pymc_bart as pmb
import arviz as az
from BayesianAnalysis.Conf import TARGET_ACC, N_TUNE, N_CORE, N_SAMPLES, N_CHAINS, BAYESIAN_RESULTS_MODEL_PRE_PATH, \
    SAMPLES_VI, XS_VALUES, BAYESIAN_RESULTS_PDP_AFTER_PATH, BAYESIAN_RESULTS_PPC_PATH, BAYESIAN_RESULTS_VI_PATH, \
    BAYESIAN_RESULTS_MODEL_AFTER_PATH, BAYESIAN_RESULTS_MODEL_POST_PATH, BAYESIAN_RESULTS_PATH, BAYESIAN_RESULTS_CONVERGENCE_PATH

from Utils.VisualizatioStyle import myStyle

myStyle()

RANDOM_SEED = 1945
np.random.seed(RANDOM_SEED)
import cloudpickle as cpkl

results_path = "F:\\users\\prasetia\\data\\Children\\children_sync\\data\\"

time = "pre"
segment = "discussion"


# time = "post"
# segment = "re3_group_climate_all"


n_model = 200


if time=="after":
    BAYESIAN_RESULTS = BAYESIAN_RESULTS_MODEL_AFTER_PATH
elif time=="post":
    BAYESIAN_RESULTS = BAYESIAN_RESULTS_MODEL_POST_PATH
else:
    BAYESIAN_RESULTS = BAYESIAN_RESULTS_MODEL_PRE_PATH

label_prefix = "_20_without_friendship"
if __name__ == '__main__':
    label = str(n_model) + "m_" + segment
    # label = str(n_model) + "m_" + segment


    with open(os.path.join(BAYESIAN_RESULTS, time + "_features_BART_" + label + ".pkl"), 'rb') as handle:
        idata = cpkl.load(handle)

    pmb.plot_convergence(idata, var_name="mu")
    plt.savefig(os.path.join(BAYESIAN_RESULTS_CONVERGENCE_PATH, label + ".pdf"), format='pdf')







