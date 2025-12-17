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
from Utils.Conf import DATA_PATH, SUMMARY_PKL_REMOVENAN_PATH
from Utils.VisualizatioStyle import myStyle
import pickle
import bambi as bmb
from scipy import stats

myStyle()
RANDOM_SEED = 1945
import cloudpickle as cpkl

feeder = DataFeeder(DATA_PATH)

extracted_columns = [
    # predictor
    "trigger_story_mean", "trigger_discussion_mean", "trigger_story_std", "trigger_discussion_std",
    # response
    "re3_group_climate_all",

    # control
    "re_gender",
    "re_background",
    "receiver_smile_occurence",

    "em_gender",
    "em_background",
    "emitter_smile_occurence",

    "group_id",
    "validator_group",
    "group_count",

]
df = feeder.fetchPostData(fill_in=False, extracted_columns=extracted_columns, file_path=SUMMARY_PKL_REMOVENAN_PATH)

predictors = [
    "trigger_story_mean",
    "trigger_story_std",
    "trigger_discussion_mean",
    "trigger_discussion_std",

]

if __name__ == '__main__':
    responseses = ["re3_group_climate_all"]
    for response in responseses:
        df[response] = np.round(df[response].values, 0) - 1
        df["re3_group_climate_all"] = pd.Categorical(
            df["re3_group_climate_all"],
            ordered=True
        )

        # normalization
        cols_to_norm = predictors
        df[cols_to_norm] = df[cols_to_norm].apply(stats.zscore)

        # Formula
        formula = """
        re3_group_climate_all ~
            trigger_story_mean +
            trigger_discussion_mean +
            trigger_story_std +
            trigger_discussion_std +
            re_gender + re_background + receiver_smile_occurence +
            em_gender + em_background + emitter_smile_occurence +
            group_count +
            (1|group_id) + (1|validator_group)
        """


        # Fit Bambi model with Ordered Logistic
        priors = {
            "Intercept": bmb.Prior("Normal", mu=0, sigma=0.5),
            "common": bmb.Prior("Normal", mu=0, sigma=0.5),  # fixed effects
            "groupsd": bmb.Prior("HalfNormal", sigma=0.5),  # SDs of random intercepts
            "sigma": bmb.Prior("HalfNormal", sigma=0.5),  # residual SD on z-scored y
            "random": bmb.Prior("Normal", mu=0, sigma=0.1),

        }
        model = bmb.Model(
            formula=formula,
            data=df,
            priors=priors,
            family="cumulative"  # ordered logistic
        )

        idata = model.fit(
            random_seed=RANDOM_SEED, target_accept=TARGET_ACC, idata_kwargs={"log_likelihood": True},
            draws=N_SAMPLES,
            chains=N_CHAINS, tune=N_TUNE, cores=N_CORE
        )

        with open(os.path.join(BAYESIAN_RESULTS_MODEL_POST_PATH, "linear_reg_post.pkl"),
                  'wb') as handle:
            pickle.dump(idata, handle, protocol=pickle.HIGHEST_PROTOCOL)