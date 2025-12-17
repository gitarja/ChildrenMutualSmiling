import matplotlib.pyplot as plt
import arviz as az
from BayesianAnalysis.Conf import BAYESIAN_RESULTS_MEAN_DIFF_PATH, BAYESIAN_RESULTS_MODEL_PATH
import pickle
from Utils.VisualizatioStyle import myStyle
import os
myStyle()



# smile occ
with open(BAYESIAN_RESULTS_MODEL_PATH + "\\" + "idata_smile_gender_story.pkl",
          'rb') as handle:
    idata_gender_story = pickle.load(handle)
with open(BAYESIAN_RESULTS_MODEL_PATH + "\\" + "idata_smile_gender_discussion.pkl",
          'rb') as handle:
    idata_gender_discussion = pickle.load(handle)


with open(BAYESIAN_RESULTS_MODEL_PATH + "\\" + "idata_smile_imigration_story.pkl",
          'rb') as handle:
    idata_immigration_story = pickle.load(handle)
with open(BAYESIAN_RESULTS_MODEL_PATH + "\\" + "idata_smile_imigration_discussion.pkl",
          'rb') as handle:
    idata_immigration_discussion = pickle.load(handle)





# smile sync
with open(BAYESIAN_RESULTS_MODEL_PATH + "\\" + "idata_smile_syc_surrogate_rate.pkl",
          'rb') as handle:
    idata_smile_sync = pickle.load(handle)

models = [idata_gender_story, idata_gender_discussion, idata_immigration_story, idata_immigration_discussion, idata_smile_sync]
az.plot_forest(models,
               model_names=["Smile occurence and gender (story)",
                            "Smile occurence and gender (discussion)",
                            "Smile occurence and immigration background (story)",
                            "Smile occurence and immigration background (discussion)",
                            "Smile sync vs surrogate"],

               var_names=["difference_of_means"],

               combined=True,
               figsize=(9, 7), hdi_prob=0.95)

# Compute summary for each model
for name, model in zip(
    ["Smile occurence and gender (story)",
     "Smile occurence and gender (discussion)",
     "Smile occurence and immigration background (story)",
     "Smile occurence and immigration background (discussion)",
     "Smile sync vs surrogate"],
    models):

    summary = az.summary(
        model,
        var_names=["difference_of_means"],
        hdi_prob=0.95
    )
    median = summary["mean"].values[0]  # ArviZ summary returns mean, sd, hdi_2.5%, hdi_97.5%
    hdi_low = summary["hdi_2.5%"].values[0]
    hdi_high = summary["hdi_97.5%"].values[0]

    print(f"{name}: median={median:.2f}, 95% HDI=({hdi_low:.2f}, {hdi_high:.2f})")

# plt.show()
plt.savefig(os.path.join(BAYESIAN_RESULTS_MEAN_DIFF_PATH,  "mean_diff_exp.pdf"), format='pdf', transparent=True,
               bbox_inches='tight')
