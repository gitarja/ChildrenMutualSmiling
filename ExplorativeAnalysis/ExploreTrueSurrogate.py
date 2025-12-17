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
import pickle
import arviz as az
from Utils.SyncMetrics import groupPoints
from scipy import stats
def getData(n_surrogate = 10, read= True):
    if read:
        df = pd.read_pickle(os.path.join(BAYESIAN_RESULTS_PATH, "surrogate_data_comparison.pkl"))
        return df
    else:
        results_path = "F:\\users\\prasetia\\data\\Children\\children_sync\\data\\"
        reader = DataReader(results_path=results_path)

        dyadic_score = reader.getDyadicScore()



        # rates and groups
        rate_list = []
        type_list = []
        receiver_list = []
        experiment_segment_list = []
        smile_rate_list = []
        experiment_segment_label = ["story", "discussion"]


        list_of_groups = glob.glob(os.path.join(EVENT_PATH, "*_eventstream.csv"))

        for group in list_of_groups:
            group_num = group.split("\\")[-1].split("_")[1]
            print(group_num)
            streams, subject_ids, story_idx, discussion_idx, smile_story, smile_discussion = reader.getData(group_num)
            indices = [story_idx, discussion_idx]
            smile_indices = [smile_story, smile_discussion]
            for i_idx in range(len(indices)):
                idx = indices[i_idx]
                smile_idx = smile_indices[i_idx]

                for i in range(len(streams) - 1):
                    for j in range(i + 1, len(streams)):

                        s1 = streams[i][idx[0]:idx[1]:]
                        s2 = streams[j][idx[0]:idx[1]:]

                        s1_occurence = len(groupPoints(np.nonzero(s1 == 1)[0], within_th=5))
                        s2_occurence = len(groupPoints(np.nonzero(s1 == 1)[0], within_th=5))

                        # score
                        group_name = group_num
                        # condition_group12 = dyadic_score[
                        #     (dyadic_score["group"] == group_name) & (dyadic_score["subject1"] == subject_ids[i]) & (
                        #             dyadic_score["subject2"] == subject_ids[j])]
                        # condition_group21 = dyadic_score[
                        #     (dyadic_score["group"] == group_name) & (dyadic_score["subject1"] == subject_ids[j]) & (
                        #             dyadic_score["subject2"] == subject_ids[i])]

                        series = np.vstack([s1, s2]).T
                        if (len(np.unique(s1)) == 2) & (len(np.unique(s2)) == 2):
                            ev = EventSeries(series, taumax=int(TAU_MAX * FPS))
                            _, t_12, _, t_21 = ev.event_coincidence_analysis(*series.T, taumax=int(TAU_MAX * FPS))
                            # if (t_12 >= 0.5) | (t_21 >= 0.5):
                            #     print(t_12)
                            #     print(t_21)
                            rate_list.append(t_12)
                            rate_list.append(t_21)
                            receiver_list.append(group_name +"_" + subject_ids[j])
                            receiver_list.append(group_name +"_" + subject_ids[i])
                            smile_rate_list.append(s2_occurence)
                            smile_rate_list.append(s1_occurence)

                            type_list.extend("trigger_rate" for _ in range(2))
                            experiment_segment_list.extend(experiment_segment_label[i_idx] for _ in range(2))
                            for _ in range(n_surrogate):

                                s1_surrogate = generateSurrogate(s1)
                                s2_surrogate = generateSurrogate(s2)
                                series_surrogate1 = np.vstack([s1, s2_surrogate]).T
                                series_surrogate2 = np.vstack([s1_surrogate, s2]).T
                                # plt.plot(s1)
                                # plt.plot(s2)
                                # plt.plot(s2_surrogate)
                                # plt.show()
                                ev_surrogate12 = EventSeries(series_surrogate1, taumax=int(TAU_MAX * FPS))
                                ev_surrogate21 = EventSeries(series_surrogate2, taumax=int(TAU_MAX * FPS))
                                _, t_12_surrogate, _, _ = ev_surrogate12.event_coincidence_analysis(*series_surrogate1.T,
                                                                                        taumax=int(TAU_MAX * FPS))
                                _, _, _, t_21_surrogate = ev_surrogate21.event_coincidence_analysis(*series_surrogate2.T,
                                                                                        taumax=int(TAU_MAX * FPS))

                                # print(t_12_surrogate)
                                # print(t_21_surrogate)

                                rate_list.append(t_12_surrogate)
                                rate_list.append(t_21_surrogate)

                                receiver_list.append(group_name +"_" +subject_ids[j])
                                receiver_list.append(group_name +"_" +subject_ids[i])

                                smile_rate_list.append(s2_occurence)
                                smile_rate_list.append(s1_occurence)

                                type_list.extend("trigger_rate_surrogate" for _ in range(2))
                                experiment_segment_list.extend(experiment_segment_label[i_idx] for _ in range(2))

        smile_rate_list = np.array(smile_rate_list) / np.max(smile_rate_list)
        df = pd.DataFrame({"rate": rate_list, "type": type_list, "experiment_segment": experiment_segment_list, "receiver": receiver_list, "smile_rate":smile_rate_list.tolist()})
        df = df.dropna()
        df.to_pickle(os.path.join(BAYESIAN_RESULTS_PATH, "surrogate_data_comparison.pkl"))
        return df
if __name__ == '__main__':


        df = getData(5, read=False)
        analyzed_features = "rate"

        df[analyzed_features] = stats.zscore(df[analyzed_features].values)

        mu_m = df[analyzed_features].mean()
        mu_s = df[analyzed_features].std() * 2

        true_obv = df.loc[df["type"] == "trigger_rate"][analyzed_features].values
        surrogate_obv = df.loc[df["type"] == "trigger_rate_surrogate"][analyzed_features].values

        true_smile_rate = df.loc[df["type"] == "trigger_rate"]["smile_rate"].values
        surr_smile_rate = df.loc[df["type"] == "trigger_rate_surrogate"]["smile_rate"].values

        true_receiver_indx, true_receiver_unique_ids = pd.factorize(df.loc[df["type"] == "trigger_rate"]["receiver"])
        sur_receiver_indx, sur_receiver_unique_ids = pd.factorize(df.loc[df["type"] == "trigger_rate_surrogate"]["receiver"])

        print(np.average(true_obv))
        print(np.average(surrogate_obv))

        sigma_low = 10 ** -1
        sigma_high = 1000
        coords = {"true_receiver_ids": true_receiver_unique_ids, "sur_receiver_ids": sur_receiver_unique_ids}
        with pm.Model(coords=coords) as model:  # model specifications in PyMC3 are wrapped in a with-statement

            # intercept
            true_receiver_intercept = pm.Normal("true_receiver_intercept", 0, 0.1, dims="true_receiver_ids")
            sur_receiver_intercept = pm.Normal("sur_receiver_intercept", 0, 0.1, dims="sur_receiver_ids")

            # slope
            true_smile_slope = pm.Normal("true_smile_slope", 0, 1)
            surr_smile_slope = pm.Normal("surr_smile_slope", 0, 1)


            true_mean = pm.Normal('true_mean', mu=mu_m, sigma=mu_s)
            surrogate_mean = pm.Normal('surrogate_mean', mu=mu_m, sigma=mu_s)

            true_std = pm.Uniform("true_std", lower=sigma_low, upper=sigma_high)
            surrogate_std = pm.Uniform("surrogate_std", lower=sigma_low, upper=sigma_high)

            nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
            nu = pm.Deterministic("nu", nu_minus_one + 1)
            nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))

            lambda_1 = true_std ** -2
            lambda_2 = surrogate_std ** -2
            lower = pm.StudentT("true", nu=nu, mu=true_mean + true_receiver_intercept[true_receiver_indx] + (true_smile_slope * true_smile_rate), lam=lambda_1, observed=true_obv)
            upper = pm.StudentT("surrogate", nu=nu, mu=surrogate_mean + sur_receiver_intercept[sur_receiver_indx] + (surr_smile_slope * surr_smile_rate) , lam=lambda_2, observed=surrogate_obv)

            diff_of_means = pm.Deterministic("difference_of_means", true_mean - surrogate_mean)
            diff_of_stds = pm.Deterministic("difference_of_std", true_std - surrogate_std)
            effect_size = pm.Deterministic(
                "effect_size", diff_of_means / np.sqrt((true_std ** 2 + surrogate_std ** 2) / 2)
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
        with open(BAYESIAN_RESULTS_MODEL_PATH + "\\" + "idata_smile_syc_surrogate_" + analyzed_features + ".pkl", 'wb') as handle:
            print("write data into: " + "idata_smile_syc_surrogate" + analyzed_features + ".pkl")
            pickle.dump(idata, handle, protocol=pickle.HIGHEST_PROTOCOL)

        trace_post = az.extract(idata.posterior)
        # list
        true_mean_list = []
        surrogate_mean_list = []
        true_hdi_list = []
        surrogate_hdi_list = []
        df_means_mean_list = []
        df_means_hdi_list = []
        features_list = []

        # compute mean and HDI 95
        true_mean_data = trace_post["true_mean"].data
        surrogate_mean_data = trace_post["surrogate_mean"].data
        true_avg = np.median(true_mean_data)
        surrogate_avg = np.median(surrogate_mean_data)
        true_hdi = az.hdi(true_mean_data, hdi_prob=.95)
        surrogate_hdi = az.hdi(surrogate_mean_data, hdi_prob=.95)

        df_means_mean = np.median(trace_post["effect_size"].data)
        df_means_hdi = az.hdi(trace_post["effect_size"].data, hdi_prob=.95)

        true_mean_list.append(true_avg)
        surrogate_mean_list.append(surrogate_avg)
        true_hdi_list.append(true_hdi)
        surrogate_hdi_list.append(surrogate_hdi)
        df_means_mean_list.append(df_means_mean)
        df_means_hdi_list.append(df_means_hdi)

        summary_df = pd.DataFrame(
            {"surrogate_median": surrogate_mean_list, "surrogate_hdi": surrogate_hdi_list,
             "true_median": true_mean_list, "true_hdi": true_hdi_list,

             "df_means_median": df_means_mean_list, "df_means_hdi": df_means_hdi_list})

        summary_df.to_csv(os.path.join(BAYESIAN_RESULTS_PATH, "hdi", "smile_surrogate.csv"))