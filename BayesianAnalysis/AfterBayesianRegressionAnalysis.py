from BayesianAnalysis.DataFeeder import DataFeeder
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import pymc_bart as pmb
import arviz as az
from BayesianAnalysis.Conf import TARGET_ACC, N_TUNE, N_CORE, N_SAMPLES, N_CHAINS, BAYESIAN_RESULTS_MODEL_AFTER_PATH, \
    XS_VALUES, BAYESIAN_RESULTS_PDP_PRE_PATH, BAYESIAN_RESULTS_PPC_PATH, BAYESIAN_RESULTS_VI_PATH
from Utils.Conf import DATA_PATH
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
                     "re2_ios",
                     "re2_ios_group",
                     "em1_leadership",
                     "em1_popularity",
                     "emre_gender",
                     "emre_ethnics",
                     "re1_bullying",
                     "em1_prosociality",
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

predictors = ["em1_friendship",
                     "re2_ios",
                     "re2_ios_group",
                     "em1_leadership",
                     "em1_popularity",
                     "emre_gender",
                     "emre_ethnics",
                     "re1_bullying",
                     #"em1_prosociality",
                     "re1_extraversion",
                ]

if __name__ == '__main__':
    segments = ["story", "discussion"]
    for segment in segments:
        seg_df = df[df["experiment_segment"] == segment]
        # df = df[df["trigger_rate"] > 0]
        X = stats.zscore(seg_df[predictors])
        y = seg_df["trigger_rate"].values
        # az.plot_dist(y, kind="hist")
        # plt.show()
        y_tranformed = stats.zscore(y)

        # smile occurence
        emitter_smile_occurence = seg_df["emitter_smile_occurence"].values
        receiver_smile_occurence = seg_df["receiver_smile_occurence"].values

        y_tranformed_min = np.min(y_tranformed)
        y_transformed_min_prior = np.average(y_tranformed == y_tranformed_min)
        y_tranformed_max = np.max(y_tranformed)
        y_transformed_max_prior = np.average(y_tranformed == y_tranformed_max)

        emitter_indx, emitter_unique_ids = pd.factorize(seg_df["em_id"])
        receiver_indx, receiver_unique_ids = pd.factorize(seg_df["re_id"])
        group_indx, group_unique_ids = pd.factorize(seg_df["group_id"])
        validator_indx, validator_unique_ids = pd.factorize(seg_df["validator_group"])
        group_count_idx, group_count_unique_ids = pd.factorize(seg_df["group_count"])
        coords = {"emitter_ids": emitter_unique_ids, "receiver_ids": receiver_unique_ids, "group_ids": group_unique_ids,
                  "group_count_ids": group_count_unique_ids,
                  "validator_ids": validator_unique_ids, "obs": range(len(X))}
        # az.plot_dist(y_tranformed, rug=True)
        # plt.show()

        label = "m_" + segment
        with pm.Model(coords=coords) as model:
            slopes = pm.Normal("slopes", mu=0, sigma=1, shape=len(predictors))  # Coefficients

            # Likelihood
            mu = pm.math.dot(X, slopes)

            ## Level 2
            emitter_intercept = pm.Normal("emitter_intercept", 0, 0.05, dims="emitter_ids")
            receiver_intercept = pm.Normal("receiver_intercept", 0, 0.05, dims="receiver_ids")
            group_intercept = pm.Normal("group_intercept", 0, 0.05, dims="group_ids")
            group_count_intercept = pm.Normal("group_count_intercept", 0, 0.05, dims="group_count_ids")
            validator_intercept = pm.Normal("validator_intercept", 0, 0.05, dims="validator_ids")

            emitter_smile_occurence_slope = pm.Normal("emitter_smile_occurence_slope", 0, 0.5)
            receiver_smile_occurence_slope = pm.Normal("receiver_smile_occurence_slope", 0, 0.5)

            random_intercept = pm.Deterministic("random_intercept",
                                                emitter_intercept[emitter_indx] + receiver_intercept[receiver_indx] +
                                                group_intercept[group_indx] +
                                                validator_intercept[validator_indx] + group_count_intercept[
                                                    group_count_idx])

            random_slope = pm.Deterministic("random_slope",
                                            (emitter_smile_occurence_slope * emitter_smile_occurence) + (
                                                        receiver_smile_occurence_slope * receiver_smile_occurence))

            mu_total = pm.Deterministic("mu_total", mu + random_intercept + random_slope)

            global_sigma = pm.HalfNormal("global_sigma", 1)

            normal_dist = pm.StudentT.dist(mu=mu_total, sigma=global_sigma, nu=4)
            y = pm.Censored("y", normal_dist, lower=y_tranformed_min, upper=y_tranformed_max, observed=y_tranformed)

        with model:
            print(model.debug())
            # pm.model_to_graphviz(model).view()
            idata = pm.sample(random_seed=RANDOM_SEED, target_accept=TARGET_ACC, idata_kwargs={"log_likelihood": True},
                              draws=N_SAMPLES,
                              chains=N_CHAINS, tune=N_TUNE, cores=N_CORE)
            pm.sample_posterior_predictive(idata, extend_inferencedata=True)

        summary = az.summary(idata, round_to=2)
        print(np.average(summary["r_hat"] > 1.01) * 100)
        az.plot_ppc(idata, num_pp_samples=100, observed_rug=True)
        plt.savefig(os.path.join(BAYESIAN_RESULTS_PPC_PATH, "after_ppc_GLM_" + label + ".pdf"), format='pdf',
                    transparent=True, bbox_inches='tight')
        plt.close()

        # save the model
        with open(os.path.join(BAYESIAN_RESULTS_MODEL_AFTER_PATH, "after_features_GLM_" + label + ".pkl"), 'wb') as handle:
            print("write data into: " + "after_features_GLM_" + label + ".pkl")
            cpkl.dump(idata, handle, protocol=cpkl.DEFAULT_PROTOCOL)

        del model
        del idata
        del X
        del y
