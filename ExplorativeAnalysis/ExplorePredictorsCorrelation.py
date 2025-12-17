import matplotlib.pyplot as plt

from BayesianAnalysis.DataFeeder import DataFeeder
from BayesianAnalysis.Conf import STATISTICAL_RESULTS_PATH
import numpy as np
from Utils.Conf import DATA_PATH
from Utils.VisualizatioStyle import myStyle
from scipy import stats
import pandas as pd
import seaborn as sn
import os
myStyle()

RANDOM_SEED = 1945
np.random.seed(RANDOM_SEED)
import cloudpickle as cpkl


feeder = DataFeeder(DATA_PATH)

extracted_columns = [
                     # predictors
                     "em1_friendship",
                     "re1_ios",
                     "re1_ios_group",
                     "em1_leadership",
                     "em1_popularity",
                     "emre_gender",
                     "emre_ethnics",
                     "re1_bullying",

                     "re1_extraversion"]

df = feeder.fetchData(only_read=True, fill_in=False, extracted_columns=extracted_columns)

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
plt.savefig(os.path.join(STATISTICAL_RESULTS_PATH,  "pre_person_correlation.pdf"), format="pdf")
# plt.show()