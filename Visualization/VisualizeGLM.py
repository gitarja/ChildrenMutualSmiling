import matplotlib.pyplot as plt
import arviz as az
from BayesianAnalysis.Conf import  BAYESIAN_RESULTS_MODEL_PRE_PATH, BAYESIAN_RESULTS_MEAN_DIFF_PATH, BAYESIAN_RESULTS_MODEL_AFTER_PATH, BAYESIAN_RESULTS_MODEL_POST_PATH
import pickle
from Utils.VisualizatioStyle import myStyle
import os
myStyle()

time = "post"
segment = "re3_group_climate_all"

# time = "pre"
# segment = "discussion"

if time == "pre":
    RESULT_PATH = BAYESIAN_RESULTS_MODEL_PRE_PATH
    filename = "linear_reg_"+segment+"_pre.pkl"
    predictors = [
        "em1_friendship",  # categorical
        "re1_ios",  # continuous
        "re1_ios_group",  # continuous
        "em1_leadership",  # continuous
        "em1_popularity",  # continuous
        "emre_gender",  # categorical
        "emre_ethnics",  # categorical
        "re1_bullying",  # continuous
        "re1_extraversion",  # continuous
    ]
    height = len(predictors) * (4.5 / 10)
elif time == "after":
    RESULT_PATH = BAYESIAN_RESULTS_MODEL_AFTER_PATH
    filename = "linear_reg_"+segment+"_pre.pkl"
    predictors = [
        "em1_friendship",  # categorical
        "re1_ios",  # continuous
        "re1_ios_group",  # continuous
        "em1_leadership",  # continuous
        "em1_popularity",  # continuous
        "emre_gender",  # categorical
        "emre_ethnics",  # categorical
        "re1_bullying",  # continuous
        "re1_extraversion",  # continuous
    ]
    height = len(predictors) * (4.5 / 10)
else:
    RESULT_PATH =  BAYESIAN_RESULTS_MODEL_POST_PATH
    filename = "linear_reg_post.pkl"
    predictors = [
        "trigger_story_mean",
        "trigger_story_std",
        "trigger_discussion_mean",
        "trigger_discussion_std",

    ]
    height = len(predictors) * (3.5 / 10)
# smile occ

with open(os.path.join(RESULT_PATH, filename), 'rb') as handle:
    idata = pickle.load(handle)


# Get the fixed-effect coefficient names from beta's coord
az.plot_forest(
    idata,
    figsize=(6, height),
    var_names=predictors,
    combined=True,
    hdi_prob=0.95,
)
# em_vars = [v for v in idata.posterior.data_vars if "em1_friendship" in v]
# az.plot_forest(idata, var_names=em_vars, combined=True, hdi_prob=0.95)
plt.axvline(x=0, color='#000000', linestyle='--', linewidth=1)
results_path = "F:\\users\\prasetia\\Personal-OneDrive\\OneDrive\\ExperimentResults\\ChildrenSync\\Final2\\Supp\\"
# plt.show()
plt.savefig(os.path.join(results_path, time+"_"+segment+".pdf"), format='pdf', transparent=True, bbox_inches='tight')