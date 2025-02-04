import matplotlib.pyplot as plt
import numpy as np
from Utils.Visualization import plotGraph
from Utils.DataReader import DataReader
from Utils.SyncMetrics import computeAdjacencyMetrix
import pandas as pd
from pyunicorn.eventseries import EventSeries
from Utils.Conf import FPS
import seaborn as sns
import glob
import os
from Utils.Conf import EVENT_PATH
from scipy.ndimage import label
from BayesianAnalysis.Conf import N_SAMPLES, N_TUNE, N_CHAINS, N_CORE, TARGET_ACC, BAYESIAN_RESULTS_PATH
import arviz as az
import pymc as pm

def readData(calculate=False):
    if calculate:

        results_path = "F:\\users\\prasetia\\data\\Children\\children_sync\\data\\"
        reader = DataReader(results_path=results_path)

        dyadic_score = reader.getDyadicScore()

        tau_max_candidate = [5]

        # rates and groups
        experiment_segment_list = []
        experiment_segment_label = ["story", "discussion"]

        duration_list = []
        percentage_list = []
        group_list = []

        list_of_groups = glob.glob(os.path.join(EVENT_PATH, "*_eventstream.csv"))

        for group in list_of_groups:
            group_num = group.split("\\")[-1].split("_")[1]
            print(group_num)
            streams, subject_ids, story_idx, discussion_idx, smile_story, smile_discussion = reader.getData(group_num)
            indices = [story_idx, discussion_idx]
            for i_idx in range(len(indices)):
                idx = indices[i_idx]
                for i in range(len(streams)):
                    s1 = streams[i][idx[0]:idx[1]:]
                    valley_groups, num_groups = label(s1)
                    duration_stream = []
                    # if np.average(s1 == 1) == 0:
                    #     print(
                    #         "Error group " + str(group_num) + ":" + str(subject_ids[i]) + "_" + experiment_segment_label[i_idx])
                    # else:
                    if np.average(s1 == 1) != 0:
                        for j in np.unique(valley_groups)[1:]:
                            point_group = np.argwhere(valley_groups == j)
                            duration = np.max(point_group) - np.min(point_group)
                            duration_stream.append(duration / FPS)

                        percentage = np.average(s1 == 1) * 100
                        duration_list.append(np.average(duration_stream))
                    else:
                        percentage = 0
                        duration_list.append(0)
                    # add experiment list
                    percentage_list.append(percentage)
                    experiment_segment_list.append(experiment_segment_label[i_idx])
                    group_list.append(group_num)

        df = pd.DataFrame(
            {"duration(sec)": duration_list, "percentage": percentage_list,
             "label": experiment_segment_list, "group": group_list})
        df.to_csv(os.path.join(BAYESIAN_RESULTS_PATH, "smile_duration.csv"))
    else:
        df = pd.read_csv(os.path.join(BAYESIAN_RESULTS_PATH, "smile_duration.csv"))

    return df


# palette = ['#252525', '#969696']
# my_palette = sns.color_palette(palette, 14)
# # shows the duration
# order = df["experiment_segment"].unique().tolist()
#
# # Plot the orbital period with horizontal boxes
# sns.boxplot(
#     df, x="experiment_segment", y="percentage",
#     whis=[0, 100], width=.6, palette=my_palette, hue="experiment_segment",fill=False
# )
#
# # Add in points to show each observation
# sns.stripplot(df, x="experiment_segment", y="percentage", size=4, color=".3", alpha=.5)
#
# plt.show()
#
# # shows the duration
# # Plot the orbital period with horizontal boxes
# sns.boxplot(
#     df, x="experiment_segment", y="duration(sec)",fill=False,
#     whis=[0, 100], width=.6, palette=my_palette, hue="experiment_segment"
# )
#
# # Add in points to show each observation
# sns.stripplot(df, x="experiment_segment", y="duration(sec)", size=4, color=".3", alpha=.5)
# plt.show()
if __name__ == '__main__':
    df = readData(False)
    group_story_idx, group_story_unique = pd.factorize(df.loc[df["label"] == "story"]["group"])
    group_discussion_idx, group_discussion_unique = pd.factorize(df.loc[df["label"] == "discussion"]["group"])
    coords = {"group_story_idx": group_story_unique, "group_discussion_idx": group_discussion_unique}

    mu_m = np.average(df["duration(sec)"].values)
    story_obv = df.loc[df["label"] == "story"]["duration(sec)"].values
    discussion_obv = df.loc[df["label"] == "discussion"]["duration(sec)"].values
    with pm.Model(coords=coords) as model:  # model specifications in PyMC3 are wrapped in a with-statement


        #random intercept
        group_story_intercept = pm.Normal("group_story_intercept", 0, 1, dims="group_story_idx")
        group_discussion_intercept = pm.Normal("group_discussion_intercept", 0, 1, dims="group_discussion_idx")
        story_mean = pm.Normal('story_mean', mu=mu_m, sigma=1)
        discussion_mean = pm.Normal('discussion_mean', mu=mu_m, sigma=1)

        story_std = pm.Uniform("story_std", lower=0.1, upper=1000)
        discussion_std = pm.Uniform("discussion_std", lower=0.1, upper=1000)


        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
        nu = pm.Deterministic("nu", nu_minus_one + 1)
        nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))

        lambda_1 = story_std ** -2
        lambda_2 = discussion_std ** -2
        story = pm.StudentT("story", nu=nu,
                                  mu=story_mean + group_story_intercept[group_story_idx],lam=lambda_1,
                                  observed=story_obv)
        discussion = pm.StudentT("discussion", nu=nu, mu=discussion_mean + group_discussion_intercept[group_discussion_idx],
                                lam=lambda_2, observed=discussion_obv)

        diff_of_means = pm.Deterministic("difference_of_means", story_mean - discussion_mean)
        diff_of_stds = pm.Deterministic("difference_of_stds", story_std - discussion_std)
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

        # print loo
    hierarchical_loo = az.plot_ppc(idata, kind='cumulative')
    plt.savefig(os.path.join(BAYESIAN_RESULTS_PATH, "PPC\\smile_duration.png"), format='png')


    trace_post = az.extract(idata.posterior)

    # list
    discussion_mean_list = []
    story_mean_list = []
    discussion_hdi_list = []
    story_hdi_list = []
    effect_size_mean_list = []
    effect_size_hdi_list = []
    features_list = []

    # compute mean and HDI 95
    discussion_mean_data = trace_post["discussion_mean"].data
    story_mean_data = trace_post["story_mean"].data
    discussion_avg = np.mean(discussion_mean_data)
    story_avg = np.mean(story_mean_data)
    discussion_hdi = az.hdi(discussion_mean_data, hdi_prob=.95)
    story_hdi = az.hdi(story_mean_data, hdi_prob=.95)

    effect_size_mean = np.mean(trace_post["effect_size"].data)
    effect_size_hdi = az.hdi(trace_post["effect_size"].data, hdi_prob=.95)

    discussion_mean_list.append(discussion_avg)
    story_mean_list.append(story_avg)
    discussion_hdi_list.append(discussion_hdi)
    story_hdi_list.append(story_hdi)
    effect_size_mean_list.append(effect_size_mean)
    effect_size_hdi_list.append(effect_size_hdi)

    summary_df = pd.DataFrame(
        {"story_mean": story_mean_list, "story_hdi": story_hdi_list,"discussion_mean": discussion_mean_list, "discussion_hdi": discussion_hdi_list,

         "effect_size_mean": effect_size_mean_list, "effect_size_hdi": effect_size_hdi_list})

    summary_df.to_csv(os.path.join(BAYESIAN_RESULTS_PATH, "hdi", "smile_duration.csv"))