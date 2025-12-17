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
sorted_probs = np.sort(status_avg_list)
cdf = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
plt.plot(sorted_probs, cdf)
# df = pd.DataFrame({"Acceptance rate (%)": status_avg_list})
# sns.stripplot(
#     data=df, x="Acceptance rate (%)",
#     dodge=True,  legend=False, size=4, color=".3"
# )
# sns.boxplot(data=df,  x="Acceptance rate (%)",  fill=False, gap=.1, showfliers=False, width=.6)

plt.ylabel("Cumulative distribution function")
plt.xlabel("Acceptance rate")
plt.show()
#print(np.average(np.concatenate(status_list) == 1))


