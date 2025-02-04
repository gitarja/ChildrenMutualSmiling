import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Utils.DataReader import DataReader
import glob
import os
from Utils.Conf import EVENT_PATH
from Utils.SyncMetrics import groupPoints
from Utils.Conf import FPS
import seaborn as sns
results_path = "F:\\users\\prasetia\\data\\Children\\children_sync\\data\\"
reader = DataReader(results_path=results_path)

dyadic_score = reader.getDyadicScore()

# rates and groups
rate_list = []
type_list = []
tau_max_list = []
experiment_segment_list = []
experiment_segment_label = ["story", "discussion"]

list_of_groups = glob.glob(os.path.join(EVENT_PATH, "*_eventstream.csv"))
overlap_rate_list = []
label_list = []
for group in list_of_groups:
    group_num = group.split("\\")[-1].split("_")[1]
    print(group_num)
    streams, subject_ids, story_idx, discussion_idx, smile_story, smile_discussion = reader.getData(group_num)
    indices = [story_idx, discussion_idx]
    smile_indices = [smile_story, smile_discussion]
    for i_idx in range(len(indices)):
        idx = indices[i_idx]
        smile_idx = smile_indices[i_idx]

        summed_stream = np.zeros_like(streams[0][idx[0]:idx[1]:])
        streams_num = []
        for i in range(len(streams)):
            summed_stream += streams[i][idx[0]:idx[1]:]
            streams_num.append(len(smile_idx[i]))

        summed_stream = (summed_stream >= 2)
        overlap_episodes = groupPoints(np.nonzero(summed_stream == 1)[0], within_th= 0.5 * FPS)

        overlap_rate = len(overlap_episodes) / np.max(streams_num)

        overlap_rate = overlap_rate if overlap_rate <= 1 else  1

        # print(overlap_rate)

        overlap_rate_list.append(overlap_rate * 100)
        label_list.append(experiment_segment_label[i_idx])

df = pd.DataFrame({"Overlap rate (%)": overlap_rate_list, "label": label_list})
sns.boxplot(data=df, x="label", y="Overlap rate (%)", color="#252525", fill=False, gap=.1, showfliers=False)
sns.stripplot(
    data=df, x="label", y="Overlap rate (%)",
    dodge=True, alpha=.2, legend=False, color="#252525",
)
plt.show()

