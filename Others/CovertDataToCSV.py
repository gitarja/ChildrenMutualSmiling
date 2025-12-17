from BayesianAnalysis.DataFeeder import DataFeeder
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import pymc_bart as pmb
import arviz as az
from BayesianAnalysis.Conf import TARGET_ACC, N_TUNE, N_CORE, N_SAMPLES, N_CHAINS, BAYESIAN_RESULTS_MODEL_PRE_PATH, \
    SAMPLES_VI, XS_VALUES, BAYESIAN_RESULTS_PDP_PRE_PATH, BAYESIAN_RESULTS_PPC_PATH, BAYESIAN_RESULTS_VI_PATH
from Utils.Conf import DATA_PATH, SUMMARY_PKL_REMOVENAN_PATH
from scipy import stats
import matplotlib as mpl
from cycler import cycler
import os
from Utils.VisualizatioStyle import myStyle

myStyle()

RANDOM_SEED = 1945
np.random.seed(RANDOM_SEED)
import cloudpickle as cpkl

feeder = DataFeeder(DATA_PATH)

extracted_columns = ["experiment_segment",
                     # predictors
                     "em1_friendship",
                     "re1_ios",
                     "re1_ios_group",
                     "em1_leadership",
                     "em1_popularity",
                     "emre_gender",
                     "emre_ethnics",
                     "re1_bullying",
                     "re1_extraversion",

                     # output
                     "trigger_rate",

                     # control
                     "em_id",
                     "re_id",
                     "seat_id",
                     "group_id",
                     "validator_group",
                     "group_count",
                     "emitter_smile_occurence",
                     "receiver_smile_occurence"

                     ]

df = feeder.fetchData(only_read=True, fill_in=False, extracted_columns=extracted_columns, file_path=SUMMARY_PKL_REMOVENAN_PATH)

df.to_csv(os.path.join("F:\\users\\prasetia\\data\\Children\\children_sync\\data\\", "SUMMARY_PKL_REMOVENAN_PATH.csv"))