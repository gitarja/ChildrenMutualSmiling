from BayesianAnalysis.DataFeeder import DataFeeder
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import pymc_bart as pmb
import arviz as az
from BayesianAnalysis.Conf import TARGET_ACC, N_TUNE, N_CORE, N_SAMPLES, N_CHAINS, BAYESIAN_RESULTS_MODEL_POST_PATH, \
    XS_VALUES, SAMPLES_VI, BAYESIAN_RESULTS_VI_PATH, BAYESIAN_RESULTS_PPC_PATH
from scipy import stats
from Utils.VisualizatioStyle import myStyle

myStyle()
RANDOM_SEED = 1945
import cloudpickle as cpkl

results_path = "F:\\users\\prasetia\\data\\Children\\children_sync\\data\\"
feeder = DataFeeder(results_path)

extracted_columns = [
    # predictor
    "trigger_story_mean", "trigger_discussion_mean", "trigger_story_std", "trigger_discussion_std",
    # response
    "re3_group_emotion_pos",
"re3_group_emotion_neg",
"re3_group_climate_collaboration",
    # control
    "re_gender", "re_background",
    "group_id", "validator_group", "group_count"]

df = feeder.fetchPostData(fill_in=False, extracted_columns=extracted_columns)

predictors = ["trigger_story_mean",
            "trigger_discussion_mean",
            "trigger_story_std",
            "trigger_discussion_std",
              "re_gender",
            "re_background"]



if __name__ == '__main__':
    responseses = [ "re3_group_emotion_neg", "re3_group_climate_collaboration"]
    for response in responseses:


        X = df[predictors]
        y = df[response]

        y_tranformed = y

        df["y"] = y_tranformed



        scale_indx, scale_unique_ids = pd.factorize(df["y"])
        K = len(scale_unique_ids)
        group_count_indx, group_count_unique_ids = pd.factorize(df["group_count"])
        group_indx, group_unique_ids = pd.factorize(df["group_id"])
        validator_indx, validator_unique_ids = pd.factorize(df["validator_group"])
        gender_indx, gender_unique_ids = pd.factorize(df["re_gender"])
        coords = {"group_ids": group_unique_ids,
                  "validator_ids": validator_unique_ids, "scales": scale_unique_ids, "group_counts": group_count_unique_ids,
                  "genders": gender_unique_ids,
                  "obs": range(len(X))}
        # az.plot_dist(y_tranformed, rug=True)
        # plt.show()
        n_model = SAMPLES_VI
        label = str(n_model) + "m_" + response
        with pm.Model(coords=coords) as model:
            mu = pmb.BART("mu", X, y_tranformed, m=n_model)

            ## Level 2
            group_intercept = pm.Normal("group_intercept", 0, 0.05, dims="group_ids")
            group_count_intercept = pm.Normal("group_count_intercept", 0, 0.05, dims="group_counts")
            validator_intercept = pm.Normal("validator_intercept", 0, 0.05, dims="validator_ids")
            gender_intercept = pm.Normal("gender_intercept", 0, 0.05, dims="genders")

            intercept = pm.Deterministic("random_intercept", group_intercept[group_indx] +
                                         validator_intercept[validator_indx] + group_count_intercept[group_count_indx] +
                                         gender_intercept[gender_indx])

            mu_total = pm.Deterministic("mu_total", mu + intercept)


            cutpoints = pm.Normal("cutpoints", mu=np.linspace(0, K, K-1), sigma=0.5,
                                  transform=pm.distributions.transforms.univariate_ordered)
            observed = pm.OrderedLogistic("observed", eta=mu_total, cutpoints=cutpoints, observed=y_tranformed-1)

        with model:
            print(model.debug())
            # pm.model_to_graphviz(model).view()
            idata = pm.sample(random_seed=RANDOM_SEED, target_accept=TARGET_ACC, idata_kwargs={"log_likelihood": True},
                              draws=N_SAMPLES,
                              chains=N_CHAINS, tune=N_TUNE, cores=N_CORE)
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=RANDOM_SEED)

            az.plot_ppc(idata, num_pp_samples=100, kind="cumulative", observed_rug=True)
            plt.savefig(BAYESIAN_RESULTS_PPC_PATH + "\\post_ppc_"+label+".pdf", format='pdf', transparent=True, bbox_inches='tight')
            plt.close()
            vi_results = pmb.compute_variable_importance(idata, mu, X, random_seed=RANDOM_SEED, samples=SAMPLES_VI,
                                                         method="VI")
            pmb.plot_variable_importance(vi_results)
            plt.savefig(BAYESIAN_RESULTS_VI_PATH + "\\post_vi_"+label+".pdf", format='pdf', transparent=True, bbox_inches='tight')
            plt.close()

            # save features importance
            with open(BAYESIAN_RESULTS_MODEL_POST_PATH + "\\" + "post_features_BART_VI_" + label + ".pkl", 'wb') as handle:
                VI_summary = vi_results
                cpkl.dump(VI_summary, handle, protocol=cpkl.DEFAULT_PROTOCOL)
                # save the model
            summary = az.summary(idata, round_to=2)
            print(np.average(summary["r_hat"] > 1.01) * 100)
            with open(BAYESIAN_RESULTS_MODEL_POST_PATH + "\\" + "post_features_BART_" + label + ".pkl", 'wb') as handle:
                print("write data into: " + "post_features_BART_" + label + ".pkl")
                cpkl.dump(idata, handle, protocol=cpkl.DEFAULT_PROTOCOL)
            with open(BAYESIAN_RESULTS_MODEL_POST_PATH + "\\" + "post_features_X_" + label + ".pkl", 'wb') as handle:
                print("write data into: " + "post_features_X_" + label + ".pkl")
                cpkl.dump(X, handle, protocol=cpkl.DEFAULT_PROTOCOL)
            with open(BAYESIAN_RESULTS_MODEL_POST_PATH + "\\" + "post_features_y_" + label + ".pkl", 'wb') as handle:
                print("write data into: " + "post_features_y_" + label + ".pkl")
                cpkl.dump(y_tranformed, handle, protocol=cpkl.DEFAULT_PROTOCOL)
            all_trees = list(model.mu.owner.op.all_trees)
            with open(BAYESIAN_RESULTS_MODEL_POST_PATH + "\\" + "post_features_mu_" + label + ".pkl", 'wb') as handle:
                print("write data into: " + "post_features_mu_" + label + ".pkl")
                cpkl.dump(all_trees, handle, protocol=cpkl.DEFAULT_PROTOCOL)

        del model
        del idata
        del X
        del y
        del all_trees
