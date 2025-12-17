from BayesianAnalysis.DataFeeder import DataFeeder
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import pymc_bart as pmb
import arviz as az
from BayesianAnalysis.Conf import TARGET_ACC, N_TUNE, N_CORE, N_SAMPLES, N_CHAINS, BAYESIAN_RESULTS_MODEL_AFTER_PATH, \
    SAMPLES_VI, XS_VALUES, BAYESIAN_RESULTS_PPC_PATH, BAYESIAN_RESULTS_VI_PATH
from Utils.Conf import DATA_PATH
import os
from scipy import stats
from Utils.VisualizatioStyle import myStyle

myStyle()

RANDOM_SEED = 1945
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

                     # response
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

# positive persona
predictors = [  "em1_friendship",
                     "re2_ios",
                     "re2_ios_group",
                     "em1_leadership",
                     "em1_popularity",
                     "emre_gender",
                     "emre_ethnics",
                     "re1_bullying",
                     #"em1_prosociality",
                     "re1_extraversion",]



if __name__ == '__main__':
    segments = ["story", "discussion"]
    for segment in segments:

        seg_df = df[df["experiment_segment"] == segment]
        # df = df[df["trigger_rate"] > 0]
        X = seg_df[predictors]
        y = seg_df["trigger_rate"].values

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
        coords = {"emitter_ids": emitter_unique_ids, "receiver_ids": receiver_unique_ids, "group_ids": group_unique_ids,
                  "validator_ids": validator_unique_ids, "obs": range(len(X))}
        #az.plot_dist(y_tranformed, rug=True)
        #plt.show()

        for n_model in [SAMPLES_VI]:
            label = str(n_model) + "m_" + segment
            with pm.Model(coords=coords) as model:
                mu = pmb.BART("mu", X, y_tranformed, m=n_model)

                ## Level 2
                emitter_intercept = pm.Normal("emitter_intercept", 0, 0.05, dims="emitter_ids")
                receiver_intercept = pm.Normal("receiver_intercept", 0, 0.05, dims="receiver_ids")
                group_intercept = pm.Normal("group_intercept", 0, 0.05, dims="group_ids")
                validator_intercept = pm.Normal("validator_intercept", 0, 0.05, dims="validator_ids")

                random_intercept = pm.Deterministic("random_intercept",
                                             emitter_intercept[emitter_indx] + receiver_intercept[receiver_indx] +
                                             group_intercept[group_indx] +
                                             validator_intercept[validator_indx])

                emitter_smile_occurence_slope = pm.Normal("emitter_smile_occurence_slope", 0, 0.5)
                receiver_smile_occurence_slope = pm.Normal("receiver_smile_occurence_slope", 0, 0.5)

                random_slope = pm.Deterministic("random_slope",
                                                (emitter_smile_occurence_slope * emitter_smile_occurence) + (
                                                            receiver_smile_occurence_slope * receiver_smile_occurence))

                mu_total = pm.Deterministic("mu_total", mu + random_intercept + random_slope)
                global_sigma = pm.HalfNormal("global_sigma", 1)
                normal_dist = pm.StudentT.dist(mu=mu_total, sigma=global_sigma, nu=4)
                y = pm.Censored("y", normal_dist, lower=y_tranformed_min, upper=y_tranformed_max, observed=y_tranformed)

                # peak_near_zero = pm.TruncatedNormal.dist(mu=y_tranformed_min, sigma=1e-5, lower=y_tranformed_min,
                #                                          upper=y_tranformed_max)
                # peak_near_max = pm.TruncatedNormal.dist(mu=y_tranformed_max, sigma=1e-3, upper=y_tranformed_max,
                #                                         lower=y_tranformed_min)
                # # student_t_component = pm.StudentT.dist(mu=mu_total, sigma=sigma, nu=nu)
                # normal_component = pm.Normal.dist(mu=mu_total, sigma=1)
                # weights = pm.Dirichlet("weights", a=np.array([y_transformed_min_prior, 1, y_transformed_max_prior]))
                #
                # y = pm.Mixture("y", w=weights, comp_dists=[normal_component, peak_near_zero, peak_near_max],
                #                observed=y_tranformed)

            with model:
                print(model.debug())
                # pm.model_to_graphviz(model).view()
                idata = pm.sample(random_seed=RANDOM_SEED, target_accept=TARGET_ACC, idata_kwargs={"log_likelihood": True},
                                  draws=N_SAMPLES,
                                  chains=N_CHAINS, tune=N_TUNE, cores=N_CORE)
                pm.sample_posterior_predictive(idata, extend_inferencedata=True)

            summary = az.summary(idata, round_to=2)
            print(np.average(summary["r_hat"] > 1.01) * 100)
            az.plot_ppc(idata, num_pp_samples=100,  observed_rug=True)
            plt.savefig(os.path.join(BAYESIAN_RESULTS_PPC_PATH, "after_ppc_"+label+".pdf"), format='pdf', transparent=True, bbox_inches='tight')
            plt.close()
            vi_results = pmb.compute_variable_importance(idata, mu, X, random_seed=RANDOM_SEED, samples=1000,
                                                         method="backward")
            pmb.plot_variable_importance(vi_results)
            plt.savefig(os.path.join(BAYESIAN_RESULTS_VI_PATH, "after_vi_"+label+".pdf"), format='pdf', transparent=True, bbox_inches='tight')
            plt.close()

            # save features importance
            with open(os.path.join(BAYESIAN_RESULTS_MODEL_AFTER_PATH, "after_features_BART_VI_" + label + ".pkl"), 'wb') as handle:
                VI_summary = vi_results
                cpkl.dump(VI_summary, handle, protocol=cpkl.DEFAULT_PROTOCOL)
                # save the model
            with open(os.path.join(BAYESIAN_RESULTS_MODEL_AFTER_PATH ,"after_features_BART_" + label + ".pkl"), 'wb') as handle:
                print("write data into: " + "after_features_BART_" + label + ".pkl")
                cpkl.dump(idata, handle, protocol=cpkl.DEFAULT_PROTOCOL)
            with open(os.path.join(BAYESIAN_RESULTS_MODEL_AFTER_PATH ,"after_features_X_" + label + ".pkl"), 'wb') as handle:
                print("write data into: " + "after_features_X_" + label + ".pkl")
                cpkl.dump(X, handle, protocol=cpkl.DEFAULT_PROTOCOL)
            with open(os.path.join(BAYESIAN_RESULTS_MODEL_AFTER_PATH , "after_features_y_" + label + ".pkl"), 'wb') as handle:
                print("write data into: " + "after_features_y_" + label + ".pkl")
                cpkl.dump(y_tranformed, handle, protocol=cpkl.DEFAULT_PROTOCOL)
            all_trees = list(model.mu.owner.op.all_trees)
            with open(os.path.join(BAYESIAN_RESULTS_MODEL_AFTER_PATH ,"after_features_mu_" + label + ".pkl"), 'wb') as handle:
                print("write data into: " + "after_features_mu_" + label + ".pkl")
                cpkl.dump(all_trees, handle, protocol=cpkl.DEFAULT_PROTOCOL)

            del model
            del idata
            del X
            del y
            del all_trees
