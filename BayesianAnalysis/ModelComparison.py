import pymc as pm
from BayesianAnalysis.Conf import BAYESIAN_RESULTS_MODEL_PATH
import pickle
import arviz as az
import matplotlib.pyplot as plt
import cloudpickle as cpkl
model_list = {}
az.style.use("arviz-darkgrid")
# for n_model in [50, 100, 150, 200]:
#
#     label = str(n_model) + "m_discussion"
#     with open(BAYESIAN_RESULTS_MODEL_PATH + "\\" + "pre_features_BART_" + label + ".pkl", 'rb') as handle:
#         idata = pickle.load(handle)
#         model_list[label] = idata
#
# df_comp_loo = az.compare(model_list)
# az.plot_compare(df_comp_loo, insample_dev=False)
#
# plt.show()


# compare filling and remove nan

with open(BAYESIAN_RESULTS_MODEL_PATH + "\\" + "pre_features_BART_200m_discussion.pkl", 'rb') as handle:
    idata_discussion_fillin= cpkl.load(handle)
with open(BAYESIAN_RESULTS_MODEL_PATH + "\\" + "pre_features_BART_200m_story.pkl", 'rb') as handle:
    idata_story_fillin = cpkl.load(handle)

with open(BAYESIAN_RESULTS_MODEL_PATH + "\\" + "pre_features_BART_200m_discussion_removenan.pkl", 'rb') as handle:
    idata_discussion_removenan = cpkl.load(handle)
with open(BAYESIAN_RESULTS_MODEL_PATH + "\\" + "pre_features_BART_200m_story_removenan.pkl", 'rb') as handle:
    idata_story_removenan = cpkl.load(handle)

hierarchical_loo = az.loo(idata_story_fillin)
df_comp_loo = az.compare({"idata_story_fillin": idata_story_fillin, "idata_story_removenan":idata_story_removenan})
az.plot_compare(df_comp_loo, insample_dev=False)
plt.show()

df_comp_loo = az.compare({"idata_discussion_fillin": idata_discussion_fillin, "idata_discussion_removenan":idata_discussion_removenan})
az.plot_compare(df_comp_loo, insample_dev=False)
plt.show()