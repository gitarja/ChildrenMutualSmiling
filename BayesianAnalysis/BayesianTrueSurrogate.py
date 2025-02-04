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
from BayesianAnalysis.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, TARGET_ACC, BAYESIAN_RESULTS_MODEL_PATH, BAYESIAN_RESULTS_POSTERIOR_PATH
import pickle
import arviz as az

def getData(n_surrogate = 10):
    results_path = "F:\\users\\prasetia\\data\\Children\\children_sync\\data\\"
    reader = DataReader(results_path=results_path)

    dyadic_score = reader.getDyadicScore()



    # rates and groups
    rate_list = []
    type_list = []
    tau_max_list = []
    experiment_segment_list = []
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

                    # score
                    group_name = group_num
                    condition_group12 = dyadic_score[
                        (dyadic_score["group"] == group_name) & (dyadic_score["subject1"] == subject_ids[i]) & (
                                dyadic_score["subject2"] == subject_ids[j])]
                    condition_group21 = dyadic_score[
                        (dyadic_score["group"] == group_name) & (dyadic_score["subject1"] == subject_ids[j]) & (
                                dyadic_score["subject2"] == subject_ids[i])]

                    series = np.vstack([s1, s2]).T
                    if (len(np.unique(s1)) == 2) & (len(np.unique(s2)) == 2):
                        ev = EventSeries(series, taumax=int(TAU_MAX * FPS))
                        _, t_12, _, t_21 = ev.event_coincidence_analysis(*series.T, taumax=int(TAU_MAX * FPS))

                        rate_list.append(t_12)
                        rate_list.append(t_21)
                        type_list.extend("trigger_rate" for _ in range(2))
                        experiment_segment_list.extend(experiment_segment_label[i_idx] for _ in range(2))
                        for _ in range(n_surrogate):
                            s1_surrogate = generateSurrogate(s1, smile_idx[i].values)
                            s2_surrogate = generateSurrogate(s2, smile_idx[j].values)
                            series_surrogate1 = np.vstack([s1, s2_surrogate]).T
                            series_surrogate2 = np.vstack([s1_surrogate, s2]).T
                            ev_surrogate12 = EventSeries(series_surrogate1, taumax=int(TAU_MAX * FPS))
                            ev_surrogate21 = EventSeries(series_surrogate2, taumax=int(TAU_MAX * FPS))
                            _, t_12_surrogate, _, _ = ev_surrogate12.event_coincidence_analysis(*series_surrogate1.T,
                                                                                    taumax=int(TAU_MAX * FPS))
                            _, _, _, t_21_surrogate = ev_surrogate21.event_coincidence_analysis(*series_surrogate2.T,
                                                                                    taumax=int(TAU_MAX * FPS))

                            rate_list.append(t_12_surrogate)
                            rate_list.append(t_21_surrogate)

                            type_list.extend("trigger_rate_surrogate" for _ in range(2))
                            experiment_segment_list.extend(experiment_segment_label[i_idx] for _ in range(2))

    df = pd.DataFrame({"rate": rate_list, "type": type_list, "experiment_segment": experiment_segment_list})
    df = df.dropna()
    return df
if __name__ == '__main__':


        df = getData()
        analyzed_features = "rate"

        mu_m = df[analyzed_features].mean()
        mu_s = df[analyzed_features].std() * 2

        true_obv = df.loc[df["type"] == "trigger_rate"][analyzed_features].values
        surrogate_obv = df.loc[df["type"] == "trigger_rate_surrogate"][analyzed_features].values

        print(np.average(true_obv))
        print(np.average(surrogate_obv))

        sigma_low = 10 ** -1
        sigma_high = 1000
        with pm.Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
            true_mean = pm.Normal('true_mean', mu=mu_m, sigma=mu_s)
            surrogate_mean = pm.Normal('surrogate_mean', mu=mu_m, sigma=mu_s)

            true_std = pm.Uniform("true_std", lower=sigma_low, upper=sigma_high)
            surrogate_std = pm.Uniform("surrogate_std", lower=sigma_low, upper=sigma_high)

            nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
            nu = pm.Deterministic("nu", nu_minus_one + 1)
            nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))

            lambda_1 = true_std ** -2
            lambda_2 = surrogate_std ** -2
            lower = pm.StudentT("true", nu=nu, mu=true_mean, lam=lambda_1, observed=true_obv)
            upper = pm.StudentT("surrogate", nu=nu, mu=surrogate_mean, lam=lambda_2, observed=surrogate_obv)

            diff_of_means = pm.Deterministic("true-surrogate_means", true_mean - surrogate_mean)
            diff_of_stds = pm.Deterministic("true-surrogate_std", true_std - surrogate_std)
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

        az.plot_posterior(idata, var_names=["true-surrogate_means", "effect_size"], ref_val=0, grid=(2, 1))
        plt.xlabel(analyzed_features)
        # plt.show()

        plt.savefig(BAYESIAN_RESULTS_POSTERIOR_PATH + "\\" + analyzed_features + ".png")
        plt.close()

        # save the model
        with open(BAYESIAN_RESULTS_MODEL_PATH + "\\" + "idata_m3_" + analyzed_features + ".pkl", 'wb') as handle:
            print("write data into: " + "idata_ttest_" + analyzed_features + ".pkl")
            pickle.dump(idata, handle, protocol=pickle.HIGHEST_PROTOCOL)
