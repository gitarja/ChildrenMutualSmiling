import matplotlib.pyplot as plt
import pandas as pd

from Utils.DataReader import DataReader
import glob
import os
from Utils.Conf import EVENT_PATH, FPS
from Utils.SyncMetrics import groupPoints
import numpy as np
import seaborn as sns
from BayesianAnalysis.Conf import N_SAMPLES, N_TUNE, N_CHAINS, N_CORE, TARGET_ACC, BAYESIAN_RESULTS_PATH
import arviz as az
import pymc as pm


def readData(calculate=False):
    if calculate:
        results_path = "F:\\users\\prasetia\\data\\Children\\children_sync\\data\\"
        reader = DataReader(results_path=results_path)



        tau_max_candidate = [5]

        # rates and groups
        experiment_segment_list = []
        experiment_segment_label = ["story", "discussion"]

        duration_list = []
        percentage_list = []

        list_of_groups = glob.glob(os.path.join(EVENT_PATH, "*_eventstream.csv"))

        story_occurence_list = []
        discussion_occurence_list = []
        label = []
        gender_list = []
        group_list = []
        individual_info = reader.getIndividualInfo(True)
        for group in list_of_groups:
            group_num = group.split("\\")[-1].split("_")[1]
            print(group_num)
            streams, subject_ids, story_idx, discussion_idx, smile_story, smile_discussion = reader.getData(
                group_num)
            for i in range(len(streams)):
                s = streams[i]
                story_stream = s[story_idx[0]:story_idx[1]]
                discussion_stream = s[discussion_idx[0]:discussion_idx[1]]
                story_occurence_list.append(len(groupPoints(np.nonzero(story_stream == 1)[0], within_th=5)))
                discussion_occurence_list.append(len(groupPoints(np.nonzero(discussion_stream == 1)[0], within_th=5)))
                # individual info
                subject = individual_info[
                    (individual_info["group"] == group_num) & (
                            individual_info["subject"] == subject_ids[i])]
                label.append("story")
                label.append("discussion")
                gender = "female" if subject["gender"].values[0] == 0 else "male"
                gender_list.extend(gender for _ in range(2))
                group_list.extend(group_num for _ in range(2))



        occurence = np.concatenate([story_occurence_list, discussion_occurence_list])

        palette = ['#252525', '#969696']
        df = pd.DataFrame({"label": label, "smile occurence": occurence, "gender": gender_list, "group": group_list})
        df.to_csv(os.path.join(BAYESIAN_RESULTS_PATH, "smile_gender.csv"))
    else:
        df = pd.read_csv(os.path.join(BAYESIAN_RESULTS_PATH, "smile_gender.csv"))

    return df


# # shows the precussor and trigger rate
# sns.stripplot(
#     data=df, x="label", y="smile occurence", hue="gender",palette=my_palette,
#     dodge=True, alpha=.4, legend=False,
# )
# sns.pointplot(df, x="label", y="smile occurence", hue="gender", palette=my_palette,
#               errorbar=None, linestyle="none",
#               estimator="median",   marker="_", markersize=20, markeredgewidth=3, dodge=.4)
# plt.show()

if __name__ == '__main__':
    df = readData(calculate=False)
    label = "gender_discussion"
    male_condition = (df["label"] == "discussion") & (df["gender"] == "male")
    female_condition = (df["label"] == "discussion") & (df["gender"] == "female")
    group_male_idx, group_male_unique = pd.factorize(df.loc[male_condition]["group"])
    group_female_idx, group_female_unique = pd.factorize(df.loc[female_condition]["group"])
    coords = {"group_male_idx": group_male_unique, "group_female_idx": group_female_unique}

    mu_m = np.average(df["smile occurence"].values)
    male_obv = df.loc[male_condition]["smile occurence"].values
    female_obv = df.loc[female_condition]["smile occurence"].values
    with pm.Model(coords=coords) as model:  # model specifications in PyMC3 are wrapped in a with-statement


        #random intercept
        group_male_intercept = pm.Normal("group_male_intercept", 0, 1, dims="group_male_idx")
        group_female_intercept = pm.Normal("group_female_intercept", 0, 1, dims="group_female_idx")
        male_mean = pm.Normal('male_mean', mu=mu_m, sigma=1)
        female_mean = pm.Normal('female_mean', mu=mu_m, sigma=1)

        male_std = pm.Uniform("male_std", lower=0.1, upper=1000)
        female_std = pm.Uniform("female_std", lower=0.1, upper=1000)


        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
        nu = pm.Deterministic("nu", nu_minus_one + 1)
        nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))

        lambda_1 = male_std ** -2
        lambda_2 = female_std ** -2
        male = pm.StudentT("male", nu=nu,
                                  mu=male_mean + group_male_intercept[group_male_idx],lam=lambda_1,
                                  observed=male_obv)
        female = pm.StudentT("female", nu=nu, mu=female_mean + group_female_intercept[group_female_idx],
                                lam=lambda_2, observed=female_obv)

        diff_of_means = pm.Deterministic("difference_of_means", male_mean - female_mean)
        diff_of_stds = pm.Deterministic("difference_of_stds", male_std - female_std)
        effect_size = pm.Deterministic(
            "effect_size", diff_of_means / np.sqrt((male_std ** 2 + female_std ** 2) / 2)
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
    plt.savefig(os.path.join(BAYESIAN_RESULTS_PATH, "PPC\\smile_"+label+".png"), format='png')


    trace_post = az.extract(idata.posterior)

    # list
    female_mean_list = []
    male_mean_list = []
    female_hdi_list = []
    male_hdi_list = []
    effect_size_mean_list = []
    effect_size_hdi_list = []
    features_list = []

    # compute mean and HDI 95
    female_mean_data = trace_post["female_mean"].data
    male_mean_data = trace_post["male_mean"].data
    female_avg = np.mean(female_mean_data)
    male_avg = np.mean(male_mean_data)
    female_hdi = az.hdi(female_mean_data, hdi_prob=.95)
    male_hdi = az.hdi(male_mean_data, hdi_prob=.95)

    effect_size_mean = np.mean(trace_post["effect_size"].data)
    effect_size_hdi = az.hdi(trace_post["effect_size"].data, hdi_prob=.95)

    female_mean_list.append(female_avg)
    male_mean_list.append(male_avg)
    female_hdi_list.append(female_hdi)
    male_hdi_list.append(male_hdi)
    effect_size_mean_list.append(effect_size_mean)
    effect_size_hdi_list.append(effect_size_hdi)

    summary_df = pd.DataFrame(
        { "male_mean": male_mean_list, "male_hdi": male_hdi_list,"female_mean": female_mean_list, "female_hdi": female_hdi_list,

         "effect_size_mean": effect_size_mean_list, "effect_size_hdi": effect_size_hdi_list})

    summary_df.to_csv(os.path.join(BAYESIAN_RESULTS_PATH, "hdi", "smile_"+label+".csv"))