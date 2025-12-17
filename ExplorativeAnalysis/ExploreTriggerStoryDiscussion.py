import matplotlib.pyplot as plt
import numpy as np
from Utils.Visualization import plotGraph
from Utils.DataReader import DataReader
from Utils.SyncMetrics import generateSurrogate
import pandas as pd
from pyunicorn.eventseries import EventSeries
from Utils.Conf import FPS, TAU_MAX
import seaborn as sns
import glob
import os
from Utils.Conf import EVENT_PATH
import pymc as pm
from BayesianAnalysis.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, TARGET_ACC, BAYESIAN_RESULTS_MODEL_PATH, BAYESIAN_RESULTS_POSTERIOR_PATH, BAYESIAN_RESULTS_PATH
from Utils.Conf import DATA_PATH, SUMMARY_PKL_REMOVENAN_PATH
import pickle
import arviz as az
from Utils.SyncMetrics import groupPoints
from scipy import stats
from BayesianAnalysis.DataFeeder import DataFeeder

if __name__ == '__main__':
        extracted_columns = ["experiment_segment",


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
        feeder = DataFeeder(DATA_PATH)
        df = feeder.fetchData(only_read=True, fill_in=False, extracted_columns=extracted_columns,
                          file_path=SUMMARY_PKL_REMOVENAN_PATH)
        analyzed_features = "trigger_rate"

        df[analyzed_features] = stats.zscore(df[analyzed_features].values)

        mu_m = df[analyzed_features].mean()
        mu_s = df[analyzed_features].std() * 2

        story_obv = df.loc[df["experiment_segment"] == "story"][analyzed_features].values
        discussion_obv = df.loc[df["experiment_segment"] == "discussion"][analyzed_features].values

        story_smile_rate = df.loc[df["experiment_segment"] == "story"]["receiver_smile_occurence"].values
        discussion_smile_rate = df.loc[df["experiment_segment"] == "discussion"]["receiver_smile_occurence"].values

        story_receiver_indx, story_receiver_unique_ids = pd.factorize(df.loc[df["experiment_segment"] == "story"]["re_id"])
        discussion_receiver_indx, discussion_receiver_unique_ids = pd.factorize(df.loc[df["experiment_segment"] == "discussion"]["re_id"])

        group_story_idx, group_story_unique = pd.factorize(df.loc[df["experiment_segment"] == "story"]["group_id"])
        group_discussion_idx, group_discussion_unique = pd.factorize(df.loc[df["experiment_segment"] == "discussion"]["group_id"])

        print(np.average(story_obv))
        print(np.average(discussion_obv))

        sigma_low = 10 ** -1
        sigma_high = 1000
        coords = {"story_receiver_ids": story_receiver_unique_ids, "discussion_receiver_ids": discussion_receiver_unique_ids,
                  "group_story_idx": group_story_unique, "group_discussion_idx": group_discussion_unique,}
        with pm.Model(coords=coords) as model:  # model specifications in PyMC3 are wrapped in a with-statement

            # random intercept
            group_story_intercept = pm.Normal("group_story_intercept", 0, 1, dims="group_story_idx")
            group_discussion_intercept = pm.Normal("group_discussion_intercept", 0, 1, dims="group_discussion_idx")

            # intercept
            story_receiver_intercept = pm.Normal("story_receiver_intercept", 0, 0.05, dims="story_receiver_ids")
            discussion_receiver_intercept = pm.Normal("discussion_receiver_intercept", 0, 0.05, dims="discussion_receiver_ids")

            # slope
            story_smile_slope = pm.Normal("story_smile_slope", 0, 1)
            discussion_smile_slope = pm.Normal("discussion_smile_slope", 0, 1)


            story_mean = pm.Normal('story_mean', mu=mu_m, sigma=mu_s)
            discussion_mean = pm.Normal('discussion_mean', mu=mu_m, sigma=mu_s)

            story_std = pm.Uniform("story_std", lower=sigma_low, upper=sigma_high)
            discussion_std = pm.Uniform("discussion_std", lower=sigma_low, upper=sigma_high)

            nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
            nu = pm.Deterministic("nu", nu_minus_one + 1)
            nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))

            lambda_1 = story_std ** -2
            lambda_2 = discussion_std ** -2
            lower = pm.StudentT("story", nu=nu, mu=story_mean + group_story_intercept[group_story_idx] + story_receiver_intercept[story_receiver_indx] + (story_smile_slope * story_smile_rate), lam=lambda_1, observed=story_obv)
            upper = pm.StudentT("discussion", nu=nu, mu=discussion_mean + group_discussion_intercept[group_discussion_idx] + discussion_receiver_intercept[discussion_receiver_indx] + (discussion_smile_slope * discussion_smile_rate) , lam=lambda_2, observed=discussion_obv)

            diff_of_means = pm.Deterministic("difference_of_means", story_mean - discussion_mean)
            diff_of_stds = pm.Deterministic("difference_of_std", story_std - discussion_std)
            effect_size = pm.Deterministic(
                "effect_size", diff_of_means / np.sqrt((story_std ** 2 + discussion_std ** 2) / 2)
            )

            # debug and sampling
        with model:
            # debug the model
            print(model.debug())
            # pm.model_to_graphviz(model).view()
            # Inference!
            idata = pm.sample_prior_predictive()
            idata.extend(
                pm.sample(random_seed=100, target_accept=TARGET_ACC, idata_kwargs={"log_likelihood": True},
                          draws=N_SAMPLES,
                          chains=N_CHAINS, tune=N_TUNE, cores=N_CORE)
            )
            idata.extend(pm.sample_posterior_predictive(idata))


        # plt.show()

        plt.savefig(BAYESIAN_RESULTS_POSTERIOR_PATH + "\\" + analyzed_features + ".png")
        plt.close()

        # save the model
        with open(BAYESIAN_RESULTS_MODEL_PATH + "\\" + "idata_trigger_story_discussion.pkl", 'wb') as handle:
            print("write data into: " + "idata_trigger_story_discussion.pkl")
            pickle.dump(idata, handle, protocol=pickle.HIGHEST_PROTOCOL)

        trace_post = az.extract(idata.posterior)
        # list
        story_mean_list = []
        discussion_mean_list = []
        story_hdi_list = []
        discussion_hdi_list = []
        df_means_mean_list = []
        df_means_hdi_list = []
        features_list = []

        # compute mean and HDI 95
        story_mean_data = trace_post["story_mean"].data
        discussion_mean_data = trace_post["discussion_mean"].data
        story_avg = np.median(story_mean_data)
        discussion_avg = np.median(discussion_mean_data)
        story_hdi = az.hdi(story_mean_data, hdi_prob=.95)
        discussion_hdi = az.hdi(discussion_mean_data, hdi_prob=.95)

        df_means_mean = np.median(trace_post["effect_size"].data)
        df_means_hdi = az.hdi(trace_post["effect_size"].data, hdi_prob=.95)

        story_mean_list.append(story_avg)
        discussion_mean_list.append(discussion_avg)
        story_hdi_list.append(story_hdi)
        discussion_hdi_list.append(discussion_hdi)
        df_means_mean_list.append(df_means_mean)
        df_means_hdi_list.append(df_means_hdi)

        summary_df = pd.DataFrame(
            {"discussion_median": discussion_mean_list, "discussion_hdi": discussion_hdi_list,
             "story_median": story_mean_list, "story_hdi": story_hdi_list,

             "df_means_median": df_means_mean_list, "df_means_hdi": df_means_hdi_list})

        summary_df.to_csv(os.path.join(BAYESIAN_RESULTS_PATH, "hdi", "trigger_story_discussion.csv"))