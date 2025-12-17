import os.path

from Utils.Conf import DATA_PATH
from Utils.DataReader import DataReader
import seaborn as sns
import matplotlib.pyplot as plt
from BayesianAnalysis.Conf import RESULTS_PATH
sns.set_style("ticks")


reader = DataReader(results_path=DATA_PATH)
individual_info = reader.getIndividualInfo(True)

sns.histplot(individual_info, x="gender")
sns.despine()
plt.savefig(os.path.join(RESULTS_PATH, "gender.pdf"), format="pdf")
plt.close()


sns.histplot(individual_info, x="age")
sns.despine()
plt.savefig(os.path.join(RESULTS_PATH, "age.pdf"), format="pdf")
plt.close()

sns.histplot(individual_info, x="height")
sns.despine()
plt.savefig(os.path.join(RESULTS_PATH, "height.pdf"), format="pdf")
plt.close()

sns.histplot(individual_info, x="weight")
sns.despine()
plt.savefig(os.path.join(RESULTS_PATH, "weight.pdf"), format="pdf")
plt.close()

sns.histplot(individual_info, x="bmi")
sns.despine()
plt.savefig(os.path.join(RESULTS_PATH, "bmi.pdf"), format="pdf")
plt.close()

sns.histplot(individual_info, x="group_count")
sns.despine()
plt.savefig(os.path.join(RESULTS_PATH, "group_count.pdf"), format="pdf")
plt.close()
