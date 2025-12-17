from BayesianAnalysis.DataFeeder import DataFeeder
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import pymc_bart as pmb
import arviz as az
from BayesianAnalysis.Conf import TARGET_ACC, N_TUNE, N_CORE, N_SAMPLES, N_CHAINS, BAYESIAN_RESULTS_MODEL_PRE_PATH, \
    XS_VALUES, BAYESIAN_RESULTS_PDP_PRE_PATH, BAYESIAN_RESULTS_PPC_PATH, BAYESIAN_RESULTS_VI_PATH
from Utils.Conf import DATA_PATH
from scipy import stats
import pickle
import os
import bambi as bmb
from Utils.VisualizatioStyle import myStyle
from pandas.api.types import CategoricalDtype
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
                     "group_id",
                     "validator_group",
                     "group_count",
                     "emitter_smile_occurence",
                     "receiver_smile_occurence"

                     ]

df = feeder.fetchData(only_read=True, fill_in=False, extracted_columns=extracted_columns)
print(len(df))
# positive persona
# df["em1_persona"] = (df["em1_nice"] + df["em1_likability"])

predictors = [

             "em1_friendship",
              "re1_ios",
              "re1_ios_group",

              "em1_leadership",
              "em1_popularity",

              "emre_gender",
              "emre_ethnics",


              "re1_bullying",
              "re1_extraversion",

              ]

if __name__ == '__main__':
    segments = ["story", "discussion"]
    for segment in segments:
        seg_df = df[df["experiment_segment"] == segment]
        response = "trigger_rate"
        predictors = [
            "em1_friendship",  # categorical
            "re1_ios",  # continuous
            "re1_ios_group",  # continuous
            "em1_leadership",  # continuous
            "em1_popularity",  # continuous
            "emre_gender",  # categorical
            "emre_ethnics",  # categorical
            "re1_bullying",  # categorical (if that's your intent)

            "re1_extraversion",  # continuous
        ]

        controls = [
            "group_count",  # continuous
            "emitter_smile_occurence",  # continuous
            "receiver_smile_occurence",  # continuous
        ]

        random_effects = ["(1|em_id)", "(1|re_id)", "(1|group_id)", "(1|validator_group)"]

        categorical_predictors = [
            "emre_ethnics",
            "em1_friendship",
            "emre_gender",
        ]

        # --- subset (if needed) ---
        seg_df = df[df["experiment_segment"] == segment].copy()


        # --- ensure categorical dtype (no baseline comparison comes from formula, not dtype) ---
        for col in categorical_predictors:
            seg_df[col] = seg_df[col].astype("category")

        # --- scale continuous only (exclude categoricals) ---
        continuous_predictors = [
            c for c in predictors + controls
            if c not in categorical_predictors
        ]

        seg_df[response] = stats.zscore(seg_df[response])
        seg_df[continuous_predictors] = seg_df[continuous_predictors].apply(stats.zscore)

        # --- build formula ---
        # Trick: start with `0 +` to drop the intercept, so categoricals get ALL levels (no baseline).
        # Use C(var) to force categorical handling.
        cat_terms =  categorical_predictors
        cont_terms = continuous_predictors
        re_terms = random_effects

        fixed_and_random =  cont_terms + cat_terms + re_terms
        formula = f"{response} ~ " + " + ".join(fixed_and_random)
        print("FORMULA:\n", formula)  # sanity check: 0 + must be right after ~
        # --- fit the model ---
        priors = {
            "Intercept": bmb.Prior("Normal", mu=0, sigma=0.1),  # fixed effects
            "common": bmb.Prior("Normal", mu=0, sigma=0.5),  # fixed effects
            "groupsd": bmb.Prior("HalfNormal", sigma=0.5),  # SDs of random intercepts
            "random": bmb.Prior("Normal", mu=0, sigma=0.1),
            "nu": bmb.Prior("Exponential", lam=1 / 30.0)  # Student-t df ~ Exp(1/30) → favor moderate tails

        }


        model = bmb.Model(
            formula=formula,
            data=seg_df,
            priors=priors,

            family="t"  # regression
        )
        idata = model.fit(
            random_seed=RANDOM_SEED, target_accept=TARGET_ACC, idata_kwargs={"log_likelihood": True},
            draws=N_SAMPLES,
            chains=N_CHAINS, tune=N_TUNE, cores=N_CORE
        )


        with open(os.path.join(BAYESIAN_RESULTS_MODEL_PRE_PATH, "linear_reg_"+segment+"_pre.pkl"),
                  'wb') as handle:
            pickle.dump(idata, handle, protocol=pickle.HIGHEST_PROTOCOL)


        del model
        del idata