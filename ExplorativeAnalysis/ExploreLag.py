import pickle

import matplotlib.pyplot as plt
import numpy as np
from Utils.Visualization import plotGraph
from Utils.DataReader import DataReader
from Utils.SyncMetrics import generateSurrogate
import pandas as pd
from pyunicorn.eventseries import EventSeries
from Utils.Conf import FPS
import seaborn as sns
import glob
import os
from Utils.Conf import EVENT_PATH
from Utils.VisualizatioStyle import myStyle

myStyle()

# results_path = "F:\\users\\prasetia\\data\\Children\\children_sync\\data\\"
# reader = DataReader(results_path=results_path)
#
# dyadic_score = reader.getDyadicScore()
#
# tau_max_candidate = [1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5]
#
#
# # rates and groups
# rate_list = []
# type_list = []
# tau_max_list = []
# experiment_segment_list = []
# experiment_segment_label = ["story", "discussion"]
#
#
# list_of_groups = glob.glob(os.path.join(EVENT_PATH, "*_eventstream.csv"))
#
# for group in list_of_groups:
#     group_num = group.split("\\")[-1].split("_")[1]
#     print(group_num)
#     streams, subject_ids, story_idx, discussion_idx, smile_story, smile_discussion = reader.getData(group_num)
#     indices = [story_idx, discussion_idx]
#     for i_idx in range(len(indices)):
#         idx = indices[i_idx]
#         for tau_max in tau_max_candidate:
#             for i in range(len(streams) - 1):
#                 for j in range(i + 1, len(streams)):
#
#                     s1 = streams[i][idx[0]:idx[1]:]
#                     s2 = streams[j][idx[0]:idx[1]:]
#
#
#
#                     # score
#                     group_name = group_num
#                     condition_group12 = dyadic_score[
#                         (dyadic_score["group"] == group_name) & (dyadic_score["subject1"] == subject_ids[i]) & (
#                                 dyadic_score["subject2"] == subject_ids[j])]
#                     condition_group21 = dyadic_score[
#                         (dyadic_score["group"] == group_name) & (dyadic_score["subject1"] == subject_ids[j]) & (
#                                 dyadic_score["subject2"] == subject_ids[i])]
#
#                     series = np.vstack([s1, s2]).T
#                     if (len(np.unique(s1)) == 2) & (len(np.unique(s2)) == 2):
#                         ev = EventSeries(series, taumax=int(tau_max * FPS))
#                         p_12, t_12, p_21, t_21 = ev.event_coincidence_analysis(*series.T, taumax=int(tau_max * FPS))
#                         # a = ev.event_analysis_significance(method='ECA', n_surr=int(1e4),
#                         #              symmetrization='directed', window_type='retarded')
#                         # matrix_ECA = ev.event_series_analysis(method='ECA',symmetrization='directed',
#                         #                                       window_type='retarded')
#                         # t_12 = matrix_ECA[0, 1]
#                         # t_21 = matrix_ECA[1, 0]
#                         rate_list.append(p_12)
#                         rate_list.append(p_21)
#                         rate_list.append(t_12)
#                         rate_list.append(t_21)
#
#                         type_list.extend("precursor_rate" for k in range(2))
#                         type_list.extend("trigger_rate" for k in range(2))
#
#                         tau_max_list.extend(str(tau_max) for k in range(4))
#
#                         # add experiment list
#                         experiment_segment_list.extend(experiment_segment_label[i_idx] for k in range(4))
#
#
#
# df = pd.DataFrame(
#     {"rate": rate_list, "type": type_list, "tau_max(sec)": tau_max_list,
#
#      "experiment_segment": experiment_segment_list})

# df.to_pickle(os.path.join("F:\\users\\prasetia\\data\\Children\\children_sync\\results\\lag_trigger_rate.pkl"), protocol=pickle.HIGHEST_PROTOCOL)

df = pd.read_pickle(os.path.join("F:\\users\\prasetia\\data\\Children\\children_sync\\results\\lag_trigger_rate.pkl"))

palette = ['#000000', '#E69F00']
my_palette = sns.color_palette(palette, 14)
df = df[df["type"] == "trigger_rate"]
# shows the precussor and trigger rate
# sns.pointplot(df, x="tau_max(sec)", y="rate", hue="experiment_segment", palette=my_palette, order=df["tau_max(sec)"].unique().tolist(),
#               errorbar=None,
#               estimator="mean", markersize=7, linestyle="--")

# sns.stripplot(
#     data=df, x="tau_max(sec)", y="rate", hue="experiment_segment",
#     dodge=True, alpha=.2, legend=False, jitter=True, size=1, palette=palette
# )
sns.pointplot(
    data=df, x="tau_max(sec)", y="rate", hue="experiment_segment",
     linestyle="none", errorbar=('ci', 95),
     markersize=10, markeredgewidth=0,  palette=palette
)
plt.show()



