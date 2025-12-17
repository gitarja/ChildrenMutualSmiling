import matplotlib.pyplot as plt
import arviz as az
from BayesianAnalysis.Conf import BAYESIAN_RESULTS_MEAN_DIFF_PATH, BAYESIAN_RESULTS_MODEL_PATH
import pickle
from Utils.VisualizatioStyle import myStyle
import os
myStyle()



# smile occ
with open(BAYESIAN_RESULTS_MODEL_PATH + "\\" + "idata_smile_occ.pkl",
          'rb') as handle:
    idata_smile_occurece = pickle.load(handle)


with open(BAYESIAN_RESULTS_MODEL_PATH + "\\" + "idata_trigger_story_discussion.pkl",
          'rb') as handle:
    idata_trigger_story_discussion = pickle.load(handle)

models = [idata_smile_occurece, idata_trigger_story_discussion]
az.plot_forest(models,
               model_names=["Smile occurence story vs discussion", "Trigger rate story vs discussion"],

               var_names=["difference_of_means"],

               combined=True,
               figsize=(9, 7), hdi_prob=0.95)

for name, model in zip(
    ["Smile occurence story vs discussion", "Trigger rate story vs discussion"],
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


plt.savefig(os.path.join(BAYESIAN_RESULTS_MEAN_DIFF_PATH,  "mean_diff_supp.pdf"), format='pdf', transparent=True,
               bbox_inches='tight')



