from BayesianAnalysis.DataFeeder import DataFeeder
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import pymc_bart as pmb
import arviz as az
from BayesianAnalysis.Conf import TARGET_ACC, N_TUNE, N_CORE, N_SAMPLES, N_CHAINS, BAYESIAN_RESULTS_MODEL_PRE_PATH, \
    SAMPLES_VI, XS_VALUES, BAYESIAN_RESULTS_PDP_PRE_PATH, BAYESIAN_RESULTS_PPC_PATH, BAYESIAN_RESULTS_VI_PATH
from Utils.Conf import DATA_PATH, SUMMARY_PKL_REMOVENAN_PATH
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
                     "seat_id",
                     "group_id",
                     "validator_group",
                     "group_count",
                     "emitter_smile_occurence",
                     "receiver_smile_occurence"

                     ]

df = feeder.fetchData(only_read=True, fill_in=False, extracted_columns=extracted_columns, file_path=SUMMARY_PKL_REMOVENAN_PATH)
print(len(df))
# positive persona
# df["em1_persona"] = (df["em1_nice"] + df["em1_likability"])

predictors = [

    "em1_friendship",
     # "re1_ios",
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
    label_prefix = "_20_without_ios"
    for segment in segments:
        seg_df = df[df["experiment_segment"] == segment]
        # df = df[df["trigger_rate"] > 0]
        X = seg_df[predictors]
        y = seg_df["trigger_rate"].values

        y_tranformed = stats.zscore(y)

        # az.plot_dist(y_tranformed, kind="hist")
        # plt.show()

        # smile occurence
        emitter_smile_occurence = seg_df["emitter_smile_occurence"].values
        receiver_smile_occurence = seg_df["receiver_smile_occurence"].values

        y_tranformed_min = np.min(y_tranformed)
        y_transformed_min_prior = np.average(y_tranformed == y_tranformed_min)
        y_tranformed_max = np.max(y_tranformed)
        y_transformed_max_prior = np.average(y_tranformed == y_tranformed_max)

        emitter_indx, emitter_unique_ids = pd.factorize(seg_df["em_id"])
        receiver_indx, receiver_unique_ids = pd.factorize(seg_df["re_id"])
        seat_indx, seat_unique_ids = pd.factorize(seg_df["seat_id"])
        group_indx, group_unique_ids = pd.factorize(seg_df["group_id"])
        validator_indx, validator_unique_ids = pd.factorize(seg_df["validator_group"])
        group_count_idx, group_count_unique_ids = pd.factorize(seg_df["group_count"])
        coords = {"seat_ids": seat_unique_ids,
                  "emitter_ids": emitter_unique_ids, "receiver_ids": receiver_unique_ids, "group_ids": group_unique_ids,
                  "group_count_ids": group_count_unique_ids,
                  "validator_ids": validator_unique_ids, "obs": range(len(X))}
        # az.plot_dist(y_tranformed, rug=True)
        # plt.show()

        for n_model in [SAMPLES_VI]:
            label = str(n_model) + "m_" + segment + label_prefix
            with pm.Model(coords=coords) as model:
                mu = pmb.BART("mu", X, y_tranformed, m=n_model, alpha=0.5)

                ## Level 2
                seat_intercept = pm.Normal("seat_intercept", 0, 0.1, dims="seat_ids")
                emitter_intercept = pm.Normal("emitter_intercept", 0, 0.1, dims="emitter_ids")
                receiver_intercept = pm.Normal("receiver_intercept", 0, 0.1, dims="receiver_ids")
                group_intercept = pm.Normal("group_intercept", 0, 0.1, dims="group_ids")
                group_count_intercept = pm.Normal("group_count_intercept", 0, 0.1, dims="group_count_ids")
                validator_intercept = pm.Normal("validator_intercept", 0, 0.1, dims="validator_ids")

                emitter_smile_occurence_slope = pm.Normal("emitter_smile_occurence_slope", 0, 0.5)
                receiver_smile_occurence_slope = pm.Normal("receiver_smile_occurence_slope", 0, 0.5)

                random_intercept = pm.Deterministic("random_intercept",
                                                    emitter_intercept[emitter_indx] + receiver_intercept[
                                                        receiver_indx] +
                                                    group_intercept[group_indx] +
                                                    validator_intercept[validator_indx] + group_count_intercept[
                                                        group_count_idx]+ seat_intercept[seat_indx]
                                                    )

                random_slope = pm.Deterministic("random_slope",
                                                (emitter_smile_occurence_slope * emitter_smile_occurence) + (
                                                            receiver_smile_occurence_slope * receiver_smile_occurence))

                mu_total = pm.Deterministic("mu_total", mu + random_intercept + random_slope)

                global_sigma = pm.HalfNormal("global_sigma", 1)

                normal_dist = pm.StudentT.dist(mu=mu_total, sigma=global_sigma, nu=4)
                y = pm.Censored("y", normal_dist, lower=y_tranformed_min, upper=y_tranformed_max, observed=y_tranformed)

                # mu_total_0 = pm.Deterministic("mu_total_0", mu[0] + random_intercept + random_slope)
                # mu_total_1 = pm.Deterministic("mu_total_1", mu[1] + random_intercept + random_slope)
                #
                # peak_1 = pm.Normal.dist(mu=mu_total_0, sigma=1)
                # peak_1_censored = pm.Censored.dist(peak_1, lower=y_tranformed_min, upper=y_tranformed_max)
                #
                # peak_2 = pm.Normal.dist(mu=mu_total_1, sigma=1)
                # peak_2_censored = pm.Censored.dist(peak_2, lower=y_tranformed_min,
                #                                            upper=y_tranformed_max)
                #
                #
                # weights = pm.Dirichlet("weights", a=np.array([1, 1]))
                #
                # y = pm.Mixture("y", w=weights, comp_dists=[peak_1_censored, peak_2_censored],
                #                observed=y_tranformed)

            with model:
                print(model.debug())
                # pm.model_to_graphviz(model).view()
                idata = pm.sample(random_seed=RANDOM_SEED, target_accept=TARGET_ACC,
                                  idata_kwargs={"log_likelihood": True},
                                  draws=N_SAMPLES,
                                  chains=N_CHAINS, tune=N_TUNE, cores=N_CORE, compile_kwargs=dict(mode="NUMBA"))
                pm.sample_posterior_predictive(idata, extend_inferencedata=True)

            summary = az.summary(idata, round_to=2)
            print(np.average(summary["r_hat"] > 1.01) * 100)
            az.plot_ppc(idata, num_pp_samples=100, observed_rug=True)
            plt.savefig(os.path.join(BAYESIAN_RESULTS_PPC_PATH, "pre_ppc_"+label+".pdf"), format='pdf', transparent=True, bbox_inches='tight')
            plt.close()
            vi_results = pmb.compute_variable_importance(idata, mu, X, random_seed=RANDOM_SEED, samples=1000,
                                                         method="backward")
            pmb.plot_variable_importance(vi_results)
            plt.savefig(os.path.join(BAYESIAN_RESULTS_VI_PATH, "pre_vi_"+label+".pdf"), format='pdf', transparent=True, bbox_inches='tight')
            plt.close()

            # save features importance
            with open(os.path.join(BAYESIAN_RESULTS_MODEL_PRE_PATH, "pre_features_BART_VI_" + label + ".pkl"), 'wb') as handle:
                VI_summary = vi_results
                cpkl.dump(VI_summary, handle, protocol=cpkl.DEFAULT_PROTOCOL)
            # save the model
            with open(os.path.join(BAYESIAN_RESULTS_MODEL_PRE_PATH, "pre_features_BART_" + label + ".pkl"),
                      'wb') as handle:
                print("write data into: " + "pre_features_BART_" + label + ".pkl")
                cpkl.dump(idata, handle, protocol=cpkl.DEFAULT_PROTOCOL)
            with open(os.path.join(BAYESIAN_RESULTS_MODEL_PRE_PATH, "pre_features_X_" + label + ".pkl"),
                      'wb') as handle:
                print("write data into: " + "pre_features_X_" + label + ".pkl")
                cpkl.dump(X, handle, protocol=cpkl.DEFAULT_PROTOCOL)
            with open(os.path.join(BAYESIAN_RESULTS_MODEL_PRE_PATH, "pre_features_y_transformed_" + label + ".pkl"),
                      'wb') as handle:
                print("write data into: " + "pre_features_y_transformed_" + label + ".pkl")
                cpkl.dump(y_tranformed, handle, protocol=cpkl.DEFAULT_PROTOCOL)
            with open(os.path.join(BAYESIAN_RESULTS_MODEL_PRE_PATH, "pre_features_y_" + label + ".pkl"),
                      'wb') as handle:
                print("write data into: " + "pre_features_y_" + label + ".pkl")
                cpkl.dump(y, handle, protocol=cpkl.DEFAULT_PROTOCOL)
            all_trees = list(model.mu.owner.op.all_trees)
            with open(os.path.join(BAYESIAN_RESULTS_MODEL_PRE_PATH, "pre_features_mu_" + label + ".pkl"),
                      'wb') as handle:
                print("write data into: " + "pre_features_mu_" + label + ".pkl")
                cpkl.dump(all_trees, handle, protocol=cpkl.DEFAULT_PROTOCOL)

            del model
            del idata
            del X
            del y
            del y_tranformed
            del all_trees
