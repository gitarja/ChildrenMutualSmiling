import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb

from BayesianAnalysis.Conf import TARGET_ACC, N_TUNE, N_CORE, N_SAMPLES, N_CHAINS, BAYESIAN_RESULTS_MODEL_POST_PATH, \
    SAMPLES_VI, BAYESIAN_RESULTS_VI_PATH, BAYESIAN_RESULTS_PPC_PATH
from BayesianAnalysis.DataFeeder import DataFeeder
from Utils.Conf import DATA_PATH, SUMMARY_PKL_REMOVENAN_PATH, SUMMARY_PKL_REMOVENAN_PATH_02, \
    SUMMARY_PKL_REMOVENAN_PATH_05, SUMMARY_PKL_REMOVENAN_PATH_10, SUMMARY_PKL_REMOVENAN_PATH_20, \
    SUMMARY_PKL_REMOVENAN_PATH_30, SUMMARY_PKL_REMOVENAN_PATH_40
import seaborn as sns


plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Arial"

RANDOM_SEED = 1945
import cloudpickle as cpkl

feeder = DataFeeder(DATA_PATH)

extracted_columns = [
    # predictor
    "trigger_story_mean", "trigger_discussion_mean", "trigger_story_std", "trigger_discussion_std",
    "re3_group_climate_all",



]

y_features = "re3_group_climate_all"
results_path = "F:\\users\\prasetia\\data\\Children\\children_sync\\results\\bayesian_ttest\\predictors\\"

df = feeder.fetchPostData(fill_in=False, extracted_columns=extracted_columns, file_path=SUMMARY_PKL_REMOVENAN_PATH_10)
fig, ax = plt.subplots(figsize=(4, 4))
sns.histplot(data=df, x=y_features, ax=ax)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize=8)
plt.show()
# plt.savefig(os.path.join(results_path, y_features+".pdf"), format="pdf")