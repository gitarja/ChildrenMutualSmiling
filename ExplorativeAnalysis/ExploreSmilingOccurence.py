import matplotlib.pyplot as plt
import pandas as pd

from Utils.DataReader import DataReader
import glob
import os
from Utils.Conf import EVENT_PATH, FPS
from Utils.SyncMetrics import groupPoints
import numpy as np
import seaborn as sns
import pymc as pm
import arviz as az
from BayesianAnalysis.Conf import N_SAMPLES, N_TUNE, N_CHAINS, N_CORE, TARGET_ACC, BAYESIAN_RESULTS_MODEL_PATH, \
    BAYESIAN_RESULTS_PATH
from Utils.Conf import DATA_PATH
import pickle
from scipy import stats


def readData(calculate=False):
    if calculate:
        results_path = "F:\\users\\prasetia\\data\\Children\\children_sync\\data\\"
        reader = DataReader(results_path=results_path)

        dyadic_score = reader.getDyadicScore()

        tau_max_candidate = [5]

        # rates and groups
        experiment_segment_label = ["story", "discussion"]

        reader = DataReader(results_path=DATA_PATH)
        validator_info = reader.getValidationInfo()
        list_of_groups = glob.glob(os.path.join(EVENT_PATH, "*_eventstream.csv"))

        story_occurence_list = []
        discussion_occurence_list = []
        label = []
        group_list = []
        validator_list = []
        for group in list_of_groups:
            group_num = group.split("\\")[-1].split("_")[1]
            validator = validator_info[validator_info["group"] == group_num]["validator"].values
            print(group_num)
            streams, subject_ids, story_idx, discussion_idx, smile_story, smile_discussion = reader.getData(
                group_num)
            for s in streams:
                story_stream = s[story_idx[0]:story_idx[1]]
                discussion_stream = s[discussion_idx[0]:discussion_idx[1]]
                story_occurence_list.append(len(groupPoints(np.nonzero(story_stream == 1)[0], within_th=5)))
                discussion_occurence_list.append(len(groupPoints(np.nonzero(discussion_stream == 1)[0], within_th=5)))
                label.append("story")
                label.append("discussion")
                group_list.append(group_num)
                group_list.append(group_num)
                validator_list.append(validator[0])
                validator_list.append(validator[0])

        occurence = np.concatenate([story_occurence_list, discussion_occurence_list])

        df = pd.DataFrame(
            {"label": label, "smile occurence": occurence, "group": group_list, "validator": validator_list})

        df.to_csv(os.path.join(BAYESIAN_RESULTS_PATH, "smile_occurence.csv"))
    else:
        df = pd.read_csv(os.path.join(BAYESIAN_RESULTS_PATH, "smile_occurence.csv"))
    return df
    # sns.boxplot(data=df, x="label", y="smile occurence", color="#252525", fill=False, gap=.1, showfliers=False)
    # sns.stripplot(
    #     data=df, x="label", y="smile occurence",
    #     dodge=True, alpha=.2, legend=False, color="#252525",
    # )
    #
    # plt.show()


if __name__ == '__main__':
    df = readData(calculate=False)
    group_story_idx, group_story_unique = pd.factorize(df.loc[df["label"] == "story"]["group"])
    group_discussion_idx, group_discussion_unique = pd.factorize(df.loc[df["label"] == "discussion"]["group"])

    validator_story_idx, validator_story_unique = pd.factorize(
        df.loc[df["label"] == "story"]["validator"])
    validator_discussion_idx, validator_discussion_unique = pd.factorize(
        df.loc[df["label"] == "discussion"]["validator"])

    coords = {"group_story_idx": group_story_unique, "group_discussion_idx": group_discussion_unique,
              "validator_discussion_idx": validator_discussion_unique, "validator_story_idx": validator_story_unique}

    df["smile occurence"] = stats.zscore(df["smile occurence"].values)
    mu_m = np.average(df["smile occurence"].values)
    mu_sigma = np.std(df["smile occurence"].values) ** 2


    story_obv = df.loc[df["label"] == "story"]["smile occurence"].values
    discussion_obv = df.loc[df["label"] == "discussion"]["smile occurence"].values

    print(np.mean(story_obv))
    print(np.mean(discussion_obv))
    with pm.Model(coords=coords) as model:  # model specifications in PyMC3 are wrapped in a with-statement

        # random intercept
        group_story_intercept = pm.Normal("group_story_intercept", 0, 1, dims="group_story_idx")
        group_discussion_intercept = pm.Normal("group_discussion_intercept", 0, 1, dims="group_discussion_idx")

        validator_story_intercept = pm.Normal("validator_story_intercept", 0, 1, dims="validator_story_idx")
        validator_discussion_intercept = pm.Normal("validator_discussion_intercept", 0, 1,
                                                   dims="validator_discussion_idx")

        story_mean = pm.Normal('story_mean', mu=mu_m, sigma=mu_sigma)
        discussion_mean = pm.Normal('discussion_mean', mu=mu_m, sigma=mu_sigma)

        story_std = pm.Uniform("story_std", lower=0.1, upper=100)
        discussion_std = pm.Uniform("discussion_std", lower=0.1, upper=100)

        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
        nu = pm.Deterministic("nu", nu_minus_one + 1)
        nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))

        lambda_1 = story_std ** -2
        lambda_2 = discussion_std ** -2
        story = pm.StudentT("story", nu=nu,
                            mu=story_mean + group_story_intercept[group_story_idx] + validator_story_intercept[
                                validator_story_idx], lam=lambda_1,
                            observed=story_obv)
        discussion = pm.StudentT("discussion", nu=nu,
                                 mu=discussion_mean + group_discussion_intercept[group_discussion_idx] +
                                    validator_discussion_intercept[validator_discussion_idx],
                                 lam=lambda_2, observed=discussion_obv)

        diff_of_means = pm.Deterministic("difference_of_means", story_mean - discussion_mean)
        diff_of_stds = pm.Deterministic("difference_of_stds", story_std - discussion_std)
        effect_size = pm.Deterministic(
            "effect_size", diff_of_means / np.sqrt((story_std ** 2 + discussion_std ** 2) / 2)
        )

        diff_validator_01 = pm.Deterministic("diff_validator_01",
                                             0.5 * ((validator_story_intercept[0] - validator_story_intercept[1]) + (
                                                         validator_discussion_intercept[0] -
                                                         validator_discussion_intercept[1])))
        diff_validator_02 = pm.Deterministic("diff_validator_02",
                                             0.5 * ((validator_story_intercept[0] - validator_story_intercept[2]) + (
                                                     validator_discussion_intercept[0] -
                                                     validator_discussion_intercept[2])))


        diff_validator_12 = pm.Deterministic("diff_validator_12",
                                             0.5 * ((validator_story_intercept[1] - validator_story_intercept[2]) + (
                                                         validator_discussion_intercept[1] -
                                                         validator_discussion_intercept[2])))

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
    hierarchical_loo = az.plot_ppc(idata)
    plt.savefig(os.path.join(BAYESIAN_RESULTS_PATH, "PPC\\smile_occurence.png"), format='png')

    # save the model
    with open(BAYESIAN_RESULTS_MODEL_PATH + "\\" + "idata_smile_occ.pkl",
              'wb') as handle:
        print("write data into: " + "idata_smile_occ.pkl")
        pickle.dump(idata, handle, protocol=pickle.HIGHEST_PROTOCOL)

    trace_post = az.extract(idata.posterior)

    # list
    discussion_mean_list = []
    story_mean_list = []
    discussion_hdi_list = []
    story_hdi_list = []
    df_means_mean_list = []
    df_means_hdi_list = []
    features_list = []

    # compute mean and HDI 95
    discussion_mean_data = trace_post["discussion_mean"].data
    story_mean_data = trace_post["story_mean"].data
    discussion_avg = np.median(discussion_mean_data)
    story_avg = np.median(story_mean_data)
    discussion_hdi = az.hdi(discussion_mean_data, hdi_prob=.95)
    story_hdi = az.hdi(story_mean_data, hdi_prob=.95)

    df_means_mean = np.median(trace_post["difference_of_means"].data)
    df_means_hdi = az.hdi(trace_post["difference_of_means"].data, hdi_prob=.95)

    discussion_mean_list.append(discussion_avg)
    story_mean_list.append(story_avg)
    discussion_hdi_list.append(discussion_hdi)
    story_hdi_list.append(story_hdi)
    df_means_mean_list.append(df_means_mean)
    df_means_hdi_list.append(df_means_hdi)

    summary_df = pd.DataFrame(
        {"story_median": story_mean_list, "story_hdi": story_hdi_list, "discussion_median": discussion_mean_list,
         "discussion_hdi": discussion_hdi_list,

         "df_means_median": df_means_mean_list, "df_means_hdi": df_means_hdi_list})

    summary_df.to_csv(os.path.join(BAYESIAN_RESULTS_PATH, "hdi", "smile_occurence.csv"))
