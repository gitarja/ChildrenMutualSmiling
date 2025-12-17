import matplotlib.pyplot as plt

from BayesianAnalysis.DataFeeder import DataFeeder
from BayesianAnalysis.Conf import STATISTICAL_RESULTS_PATH
from Utils.VisualizatioStyle import myStyle
from Utils.Conf import DATA_PATH
import pandas as pd
import numpy as np
import os
myStyle()
RANDOM_SEED = 1945
from scipy import stats
import seaborn as sn

feeder = DataFeeder(DATA_PATH)


extracted_columns = ["trigger_story_mean",
            "trigger_discussion_mean",
            "trigger_story_std",
            "trigger_discussion_std",
            # "re_gender",
            # "re_background"
              ]
df = feeder.fetchPostData(fill_in=False, extracted_columns=extracted_columns)

corr_mat = np.zeros((len(extracted_columns), len(extracted_columns)))
for i in range(len(extracted_columns)):
    for j in range(len(extracted_columns)):
        c1 = extracted_columns[i]
        c2 = extracted_columns[j]
        data1 = df[c1].values
        data2 = df[c2].values
        res  = stats.pearsonr(data1, data2)
        corr_mat[i, j] = res.statistic

df_cm = pd.DataFrame(corr_mat, index = [i for i in extracted_columns],
                  columns = [i for i in extracted_columns])
sn.heatmap(df_cm, annot=True, cmap=sn.color_palette("Blues", as_cmap=True), fmt=".2f")
plt.savefig(os.path.join(STATISTICAL_RESULTS_PATH,  "post_person_correlation.pdf"), format="pdf")


# x = df["trigger_discussion_mean"]
# y = df["trigger_discussion_std"]
# slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
# sn.jointplot(data=df, x="trigger_discussion_mean", y="trigger_discussion_std")
# # sn.lmplot(x="trigger_story_std", y="trigger_story_mean", data=df,
# #            y_jitter=.03, scatter_kws={"s": 20, "linewidths":0})
# # Add p-value as text
# plt.text(x=min(x), y=max(y), s=f'r-value = {r_value:.2f}', fontsize=12)
# plt.show()