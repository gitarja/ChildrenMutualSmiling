import matplotlib.pyplot as plt
import pandas as pd

from Utils.DataReader import DataReader
import glob
import os
from Utils.Conf import EVENT_PATH, FPS
from Utils.SyncMetrics import groupPoints
import numpy as np
import seaborn as sns
from BayesianAnalysis.Conf import N_SAMPLES, N_TUNE, N_CHAINS, N_CORE, TARGET_ACC, BAYESIAN_RESULTS_PATH,BAYESIAN_RESULTS_MODEL_PATH
from Utils.Conf import DATA_PATH
import arviz as az
import pymc as pm
import pickle
from scipy import stats


def readData(calculate=False):
    if calculate:
        results_path = "F:\\users\\prasetia\\data\\Children\\children_sync\\data\\"
        reader = DataReader(results_path=results_path)



        tau_max_candidate = [5]

        # rates and groups
        experiment_segment_list = []
        experiment_segment_label = ["story", "discussion"]



        reader = DataReader(results_path=DATA_PATH)
        validator_info = reader.getValidationInfo()
        list_of_groups = glob.glob(os.path.join(EVENT_PATH, "*_eventstream.csv"))

        story_occurence_list = []
        discussion_occurence_list = []
        label = []
        imigration_list = []
        group_list = []
        validator_list = []
        individual_info = reader.getIndividualInfo(True)
        for group in list_of_groups:
            group_num = group.split("\\")[-1].split("_")[1]
            validator = validator_info[validator_info["group"] == group_num]["validator"].values
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
                imigration = "german" if subject["nationality"].values[0] == 1 else "non-german"
                imigration_list.extend(imigration for _ in range(2))
                group_list.extend(group_num for _ in range(2))
                validator_list.append(validator[0])
                validator_list.append(validator[0])



        occurence = np.concatenate([story_occurence_list, discussion_occurence_list])

        palette = ['#252525', '#969696']
        df = pd.DataFrame({"label": label, "smile occurence": occurence, "nationality": imigration_list, "group": group_list, "validator": validator_list})
        df.to_csv(os.path.join(BAYESIAN_RESULTS_PATH, "smile_immigration.csv"))
    else:
        df = pd.read_csv(os.path.join(BAYESIAN_RESULTS_PATH, "smile_immigration.csv"))

    return df



if __name__ == '__main__':
    df = readData(calculate=False)
    segment ="story"
    label = "imigration_"+segment
    german_condition = (df["label"] == segment) & (df["nationality"] == "german")
    non_german_condition = (df["label"] == segment) & (df["nationality"] == "non-german")
    group_german_idx, group_german_unique = pd.factorize(df.loc[german_condition]["group"])
    group_non_german_idx, group_non_german_unique = pd.factorize(df.loc[non_german_condition]["group"])

    validator_german_idx, validator_german_unique = pd.factorize(
        df.loc[german_condition]["validator"])
    validator_non_german_idx, validator_non_german_unique = pd.factorize(
        df.loc[non_german_condition]["validator"])

    coords = {"group_german_idx": group_german_unique, "group_non_german_idx": group_non_german_unique,
              "validator_german_idx": validator_german_unique, "validator_non_german_idx": validator_non_german_unique}

    df["smile occurence"] = stats.zscore(df["smile occurence"].values)

    mu_m = np.average(df["smile occurence"].values)
    mu_sigma = np.std(df["smile occurence"].values) ** 2

    print(mu_m)
    print(mu_sigma)
    german_obv = df.loc[german_condition]["smile occurence"].values
    non_german_obv = df.loc[non_german_condition]["smile occurence"].values
    with pm.Model(coords=coords) as model:  # model specifications in PyMC3 are wrapped in a with-statement


        #random intercept
        group_german_intercept = pm.Normal("group_german_intercept", 0, 1, dims="group_german_idx")
        group_non_german_intercept = pm.Normal("group_non_german_intercept", 0, 1, dims="group_non_german_idx")

        validator_german_intercept = pm.Normal("validator_german_intercept", 0, 1, dims="validator_german_idx")
        validator_non_german_intercept = pm.Normal("validator_non_german_intercept", 0, 1,
                                                   dims="validator_non_german_idx")

        german_mean = pm.Normal('german_mean', mu=mu_m, sigma=mu_sigma)
        non_german_mean = pm.Normal('non_german_mean', mu=mu_m, sigma=mu_sigma)

        german_std = pm.Uniform("german_std", lower=0.1, upper=1000)
        non_german_std = pm.Uniform("non_german_std", lower=0.1, upper=1000)


        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
        nu = pm.Deterministic("nu", nu_minus_one + 1)
        nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))

        lambda_1 = german_std ** -2
        lambda_2 = non_german_std ** -2
        german = pm.StudentT("german", nu=nu,
                                  mu=german_mean + group_german_intercept[group_german_idx] + validator_german_intercept[validator_german_idx],lam=lambda_1,
                                  observed=german_obv)
        non_german = pm.StudentT("non_german", nu=nu, mu=non_german_mean + group_non_german_intercept[group_non_german_idx] + validator_non_german_intercept[validator_non_german_idx],
                                lam=lambda_2, observed=non_german_obv)

        diff_of_means = pm.Deterministic("difference_of_means", german_mean - non_german_mean)
        diff_of_stds = pm.Deterministic("difference_of_stds", german_std - non_german_std)
        effect_size = pm.Deterministic(
            "effect_size", diff_of_means / np.sqrt((german_std ** 2 + non_german_std ** 2) / 2)
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

    # save the model
    with open(BAYESIAN_RESULTS_MODEL_PATH + "\\" + "idata_smile_imigration_"+segment+".pkl",
              'wb') as handle:
        print("write data into: " + "idata_smile_imigration_"+segment+".pkl")
        pickle.dump(idata, handle, protocol=pickle.HIGHEST_PROTOCOL)

    trace_post = az.extract(idata.posterior)

    # list
    non_german_mean_list = []
    german_mean_list = []
    non_german_hdi_list = []
    german_hdi_list = []
    df_means_mean_list = []
    df_means_hdi_list = []
    features_list = []

    # compute mean and HDI 95
    non_german_mean_data = trace_post["non_german_mean"].data
    german_mean_data = trace_post["german_mean"].data
    non_german_avg = np.median(non_german_mean_data)
    german_avg = np.median(german_mean_data)
    non_german_hdi = az.hdi(non_german_mean_data, hdi_prob=.95)
    german_hdi = az.hdi(german_mean_data, hdi_prob=.95)

    df_means_mean = np.median(trace_post["effect_size"].data)
    df_means_hdi = az.hdi(trace_post["effect_size"].data, hdi_prob=.95)

    non_german_mean_list.append(non_german_avg)
    german_mean_list.append(german_avg)
    non_german_hdi_list.append(non_german_hdi)
    german_hdi_list.append(german_hdi)
    df_means_mean_list.append(df_means_mean)
    df_means_hdi_list.append(df_means_hdi)

    summary_df = pd.DataFrame(
        {"german_median": german_mean_list, "german_hdi": german_hdi_list, "non_german_median": non_german_mean_list,
         "non_german_hdi": non_german_hdi_list,

         "df_means_median": df_means_mean_list, "df_means_hdi": df_means_hdi_list})

    summary_df.to_csv(os.path.join(BAYESIAN_RESULTS_PATH, "hdi", "smile_" + label + ".csv"))