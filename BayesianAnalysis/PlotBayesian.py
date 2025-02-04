from BayesianAnalysis.Conf import BAYESIAN_RESULTS_MODEL_PATH
import pickle
import arviz as az
import matplotlib.pyplot as plt
# save the model
analyzed_features = "rate"
with open(BAYESIAN_RESULTS_MODEL_PATH + "\\" + "idata_m3_" + analyzed_features + ".pkl", 'rb') as handle:
    idata = pickle.load(handle)

az.plot_posterior(idata, var_names=["true-surrogate_means", "effect_size"], ref_val=0, grid=(2, 1))
plt.xlabel(analyzed_features)
plt.show()

# az.plot_posterior(idata, var_names=["true_mean", "surrogate_mean", "true_std", "surrogate_std"], grid=(2, 2))
# plt.xlabel(analyzed_features)
# plt.show()
