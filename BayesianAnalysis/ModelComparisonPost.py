import pymc as pm
from BayesianAnalysis.Conf import BAYESIAN_RESULTS_MODEL_POST_PATH
import pickle
import arviz as az
import matplotlib.pyplot as plt
import cloudpickle as cpkl
model_list = {}
az.style.use("arviz-darkgrid")

# compare filling and remove nan

with open(BAYESIAN_RESULTS_MODEL_POST_PATH + "\\" + "post_features_BART_200m_story_re3_group_climate_all.pkl", 'rb') as handle:
    idata_story = cpkl.load(handle)
with open(BAYESIAN_RESULTS_MODEL_POST_PATH + "\\" + "post_features_BART_200m_discussion_re3_group_climate_all.pkl", 'rb') as handle:
    idata_discussion = cpkl.load(handle)

with open(BAYESIAN_RESULTS_MODEL_POST_PATH + "\\" + "post_features_BART_200m_re3_group_climate_all.pkl", 'rb') as handle:
    idata_full = cpkl.load(handle)



df_comp_loo = az.compare({

    "story": idata_story,
    "discussion": idata_discussion,
    "full model": idata_full

})
az.plot_compare(df_comp_loo, insample_dev=False)
plt.show()

