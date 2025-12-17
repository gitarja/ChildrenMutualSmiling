import pymc as pm
from BayesianAnalysis.Conf import BAYESIAN_RESULTS_MODEL_PRE_PATH
import pickle
import arviz as az
import matplotlib.pyplot as plt
import cloudpickle as cpkl
model_list = {}
az.style.use("arviz-darkgrid")

# compare filling and remove nan

with open(BAYESIAN_RESULTS_MODEL_PRE_PATH + "\\" + "pre_features_BART_200m_individual_expstory.pkl", 'rb') as handle:
    idata_individual_exp = cpkl.load(handle)
with open(BAYESIAN_RESULTS_MODEL_PRE_PATH + "\\" + "pre_features_BART_200m_social_demographicstory.pkl", 'rb') as handle:
    idata_demographic = cpkl.load(handle)

with open(BAYESIAN_RESULTS_MODEL_PRE_PATH + "\\" + "pre_features_BART_200m_social_statusstory.pkl", 'rb') as handle:
    idata_status = cpkl.load(handle)
with open(BAYESIAN_RESULTS_MODEL_PRE_PATH + "\\" + "pre_features_BART_200m_social_bondstory.pkl", 'rb') as handle:
    idata_social_bond = cpkl.load(handle)

with open(BAYESIAN_RESULTS_MODEL_PRE_PATH + "\\" + "pre_features_BART_200m_story.pkl", 'rb') as handle:
        idata_full = cpkl.load(handle)


df_comp_loo = az.compare({
    "individual experience": idata_individual_exp,
    "demographic":idata_demographic,
    "social status": idata_status,
    "social bonds": idata_social_bond,
    "full model": idata_full

})
az.plot_compare(df_comp_loo, insample_dev=False)
plt.show()

