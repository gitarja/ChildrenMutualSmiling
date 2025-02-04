import numpy as np

from Utils.Conf import SMILE_DETECTION_PATH
from Utils.VisualizatioStyle import myStyle
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

myStyle()
files = glob.glob(SMILE_DETECTION_PATH + "\\*.pkl")
status_avg_list = []
# accept = 1
# reject = 2
for f in files:
    smile_file = pd.read_pickle(f)
    status = smile_file["status"].values
    status_avg_list.append(np.average(status == 1) * 100)


print(np.average(status_avg_list))
print(np.std(status_avg_list))
az.plot_dist(status_avg_list, rug=True)
# df = pd.DataFrame({"Acceptance rate (%)": status_avg_list})
# sns.boxplot(data=df,  y="Acceptance rate (%)", color="#252525", fill=False, gap=.1, showfliers=False)
# sns.stripplot(
#     data=df, y="Acceptance rate (%)",
#     dodge=True, alpha=.2, legend=False, color="#252525",
# )
plt.xlabel("Acceptance rate (%)")
plt.show()
#print(np.average(np.concatenate(status_list) == 1))


