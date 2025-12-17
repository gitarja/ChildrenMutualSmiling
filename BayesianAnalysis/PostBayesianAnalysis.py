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
from Utils.Conf import DATA_PATH, SUMMARY_PKL_REMOVENAN_PATH, SUMMARY_PKL_REMOVENAN_PATH_02, SUMMARY_PKL_REMOVENAN_PATH_05, SUMMARY_PKL_REMOVENAN_PATH_10, SUMMARY_PKL_REMOVENAN_PATH_20, SUMMARY_PKL_REMOVENAN_PATH_30, SUMMARY_PKL_REMOVENAN_PATH_40


from Utils.VisualizatioStyle import myStyle

myStyle()
RANDOM_SEED = 1945
import cloudpickle as cpkl

feeder = DataFeeder(DATA_PATH)

extracted_columns = [
    # predictor
    "trigger_story_mean", "trigger_discussion_mean", "trigger_story_std", "trigger_discussion_std",
    # response
    "re3_group_climate_all",
    "em3_group_climate_all",
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

    # control

]
label_prefix = "_10"
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
        X = df[predictors]

        y = df[response].values
        y_tranformed = np.round(y, 0)


        group_count_indx, group_count_unique_ids = pd.factorize(df["group_count"])
        group_indx, group_unique_ids = pd.factorize(df["group_id"])
        validator_indx, validator_unique_ids = pd.factorize(df["validator_group"])

        if "em3_" in response:
            gender_indx, gender_unique_ids = pd.factorize(df["em_gender"])
            background_indx, background_unique_ids = pd.factorize(df["em_background"])
        else:
            gender_indx, gender_unique_ids = pd.factorize(df["re_gender"])
            background_indx, background_unique_ids = pd.factorize(df["re_background"])
        coords = {"group_ids": group_unique_ids,
                  "validator_ids": validator_unique_ids,
                  "group_counts": group_count_unique_ids,
                  "genders": gender_unique_ids,
                  "background": background_unique_ids,
                  "obs": range(len(X))}
        # print(np.unique(y_tranformed.astype(int)))
        # plt.hist(y_tranformed)
        # plt.scatter(y_tranformed, X["trigger_story_mean"])


        # plt.show()
        n_model = SAMPLES_VI
        label = str(n_model) + "m_" + response + label_prefix
        K = len(np.unique(y_tranformed))
        n_obs = len(y_tranformed)

        # smile occurence
        if "em3_" in response:
            smile_occurence = df["emitter_smile_occurence"].values
        else:
            smile_occurence = df["receiver_smile_occurence"].values


        with pm.Model(coords=coords) as model:
            mu = pmb.BART("mu", X, y_tranformed, m=n_model, alpha=0.5)

            ## Level 2
            group_intercept = pm.Normal("group_intercept", 0, 0.1, dims="group_ids")
            group_count_intercept = pm.Normal("group_count_intercept", 0, 0.1, dims="group_counts")
            validator_intercept = pm.Normal("validator_intercept", 0, 0.1, dims="validator_ids")
            gender_intercept = pm.Normal("gender_intercept", 0, 0.1, dims="genders")
            background_intercept = pm.Normal("background_intercept", 0, 0.1, dims="background")

            smile_occurence_slope = pm.Normal("smile_occurence_slope", 0, 0.5)

            intercept = pm.Deterministic("random_intercept", group_intercept[group_indx] +
                                         validator_intercept[validator_indx] + group_count_intercept[group_count_indx] +
                                         gender_intercept[gender_indx] + background_intercept[background_indx])

            mu_total = pm.Deterministic("mu_total",
                                        mu + intercept + (smile_occurence_slope * smile_occurence))

            # Most observed values are clustered on some values and the distribution is not normal.
            # Hence we considered the observed values as ordinal.
            global_sigma = pm.HalfNormal("global_sigma", 0.5)
            cutpoints = pm.Normal("cutpoints", mu=np.linspace(0, K, K - 1), sigma=global_sigma,
                                  transform=pm.distributions.transforms.univariate_ordered)
            observed = pm.OrderedLogistic("observed", eta=mu_total, cutpoints=cutpoints, observed=y_tranformed - 1)

        with model:
            print(model.debug())
            # pm.model_to_graphviz(model).view()
            idata = pm.sample(random_seed=RANDOM_SEED, target_accept=TARGET_ACC, idata_kwargs={"log_likelihood": True},
                              draws=N_SAMPLES,
                              chains=N_CHAINS, tune=N_TUNE, cores=N_CORE)

            pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=RANDOM_SEED)

            az.plot_ppc(idata, num_pp_samples=200, observed_rug=True)
            plt.savefig(os.path.join(BAYESIAN_RESULTS_PPC_PATH, "post_ppc_" + label + ".pdf"), format='pdf',
                        transparent=True, bbox_inches='tight')
            plt.close()

            vi_results = pmb.compute_variable_importance(idata, mu, X, random_seed=RANDOM_SEED, samples=1000,
                                                         method="backward")
            pmb.plot_variable_importance(vi_results)
            plt.savefig(os.path.join(BAYESIAN_RESULTS_VI_PATH, "post_vi_" + label + ".pdf"), format='pdf',
                        transparent=True, bbox_inches='tight')
            plt.close()

            # save features importance
            with open(os.path.join(BAYESIAN_RESULTS_MODEL_POST_PATH, "post_features_BART_VI_" + label + ".pkl"),
                      'wb') as handle:
                VI_summary = vi_results
                cpkl.dump(VI_summary, handle, protocol=cpkl.DEFAULT_PROTOCOL)
                # save the model
            summary = az.summary(idata, round_to=2)
            print(np.average(summary["r_hat"] > 1.01) * 100)
            with open(os.path.join(BAYESIAN_RESULTS_MODEL_POST_PATH, "post_features_BART_" + label + ".pkl"),
                      'wb') as handle:
                print("write data into: " + "post_features_BART_" + label + ".pkl")
                cpkl.dump(idata, handle, protocol=cpkl.DEFAULT_PROTOCOL)
            with open(os.path.join(BAYESIAN_RESULTS_MODEL_POST_PATH, "post_features_X_" + label + ".pkl"),
                      'wb') as handle:
                print("write data into: " + "post_features_X_" + label + ".pkl")
                cpkl.dump(X, handle, protocol=cpkl.DEFAULT_PROTOCOL)
            with open(os.path.join(BAYESIAN_RESULTS_MODEL_POST_PATH, "post_features_y_" + label + ".pkl"),
                      'wb') as handle:
                print("write data into: " + "post_features_y_" + label + ".pkl")
                cpkl.dump(y_tranformed, handle, protocol=cpkl.DEFAULT_PROTOCOL)
            all_trees = list(model.mu.owner.op.all_trees)
            with open(os.path.join(BAYESIAN_RESULTS_MODEL_POST_PATH, "post_features_mu_" + label + ".pkl"),
                      'wb') as handle:
                print("write data into: " + "post_features_mu_" + label + ".pkl")
                cpkl.dump(all_trees, handle, protocol=cpkl.DEFAULT_PROTOCOL)

        del model
        del idata
        del X
        del y
        del all_trees
