import os.path

from BayesianAnalysis.DataFeeder import DataFeeder
import numpy as np
from Utils.Conf import DATA_PATH, SUMMARY_PKL_REMOVENAN_PATH_20
from Utils.VisualizatioStyle import myStyle
import seaborn as sns
import matplotlib.pyplot as plt

RANDOM_SEED = 1945
np.random.seed(RANDOM_SEED)
import cloudpickle as cpkl


plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Arial"



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

y_features = "emitter_smile_occurence"
results_path = "F:\\users\\prasetia\\data\\Children\\children_sync\\results\\bayesian_ttest\\predictors\\"

df = feeder.fetchData(only_read=True, fill_in=False, extracted_columns=extracted_columns, file_path=SUMMARY_PKL_REMOVENAN_PATH_20)

fig, ax = plt.subplots(figsize=(4, 4))

# sns.scatterplot(data=df, x="re1_ios", y="re1_ios_group")

sns.stripplot(
    data=df, x="group_count", y=y_features, hue="group_count",
    dodge=True, alpha=.1, legend=False, size=2, ax=ax
)
sns.pointplot(
    data=df, x="group_count", y=y_features, hue="group_count",
    dodge=0.6, linestyle="none", errorbar=('ci', 95), capsize=.1,
    markersize=1.5, markeredgewidth=1.5, linewidth=0.75, palette="dark"
)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize=8)
plt.show()
# plt.savefig(os.path.join(results_path, y_features+".pdf"), format="pdf")