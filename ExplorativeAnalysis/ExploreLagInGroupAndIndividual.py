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
from Utils.Conf import EVENT_PATH, TAU_MAX

results_path = "F:\\users\\prasetia\\data\\Children\\children_sync\\data\\"
reader = DataReader(results_path=results_path)

dyadic_score = reader.getDyadicScore()

tau_max_candidate = [1, 2, 3, 4, 5, 6, 7]
max_length = 17 * 60 * 25  # 19 min * fps
fps = 25  # tau max is one second

# rates and groups
rate_list = []
type_list = []
group_id_list = []
experiment_segment_list = []
experiment_segment_label = ["story", "discussion"]

list_of_groups = glob.glob(os.path.join(EVENT_PATH, "*_eventstream.csv"))

for group in list_of_groups:
    group_num = group.split("\\")[-1].split("_")[1]
    print(group_num)
    streams, subject_ids, story_idx, discussion_idx, smile_story, smile_discussion = reader.getData(group_num)
    indices = [story_idx, discussion_idx]
    for i_idx in range(len(indices)):
        idx = indices[i_idx]

        for i in range(len(streams) - 1):
            for j in range(i + 1, len(streams)):

                s1 = streams[i][idx[0]:idx[1]:]
                s2 = streams[j][idx[0]:idx[1]:]

                # score
                group_name = group_num
                condition_group12 = dyadic_score[
                    (dyadic_score["group"] == group_name) & (dyadic_score["subject1"] == subject_ids[i]) & (
                            dyadic_score["subject2"] == subject_ids[j])]
                condition_group21 = dyadic_score[
                    (dyadic_score["group"] == group_name) & (dyadic_score["subject1"] == subject_ids[j]) & (
                            dyadic_score["subject2"] == subject_ids[i])]

                series = np.vstack([s1, s2]).T
                if (len(np.unique(s1)) == 2) & (len(np.unique(s2)) == 2):
                    ev = EventSeries(series, taumax=int(TAU_MAX * FPS))
                    _, t_12, _, t_21 = ev.event_coincidence_analysis(*series.T, taumax=int(TAU_MAX * FPS))

                    rate_list.append(t_12)
                    rate_list.append(t_21)

                    type_list.extend("trigger_rate" for k in range(2))

                    group_id_list.extend(group_num for _ in range(2))
                    # add experiment list
                    experiment_segment_list.extend(experiment_segment_label[i_idx] for k in range(2))

df = pd.DataFrame(
    {"rate": rate_list, "type": type_list,
     "group": group_id_list,
     "experiment_segment": experiment_segment_list})


# my_palette = "PuBu"
# # shows the precussor and trigger rate
# sns.stripplot(
#     data=df, x="experiment_segment", y="rate", hue="experiment_segment",palette=my_palette,
#     dodge=True, alpha=.1, legend=False,
# )
# sns.pointplot(df, x="experiment_segment", y="rate", hue="experiment_segment", palette=my_palette,
#               errorbar=None, linestyle="none",
#               estimator="median",   marker="_", markersize=20, markeredgewidth=3, dodge=.4)
# plt.show()


group_experiment_df = df.groupby(["group", "experiment_segment"])

group_trigger_list = []
group_type_trigger_list = []
group_id_list = []
experiment_segment_list = []
for _, g in group_experiment_df:
    group_trigger_list.append(np.nanmean(g["rate"].values))
    group_trigger_list.append(np.nanstd(g["rate"].values))
    group_type_trigger_list.append("mean")
    group_type_trigger_list.append("std")
    group_id_list.extend(g["group"].values[0] for _ in range(2))
    experiment_segment_list.extend(g["experiment_segment"].values[0] for _ in range(2))

group_df = pd.DataFrame(
    {"rate": group_trigger_list, "type": group_type_trigger_list,
     "group": group_id_list,
     "experiment_segment": experiment_segment_list})

palette = ['#252525', '#969696']
my_palette = sns.color_palette(palette, 14)
# shows the precussor and trigger rate
sns.stripplot(
    data=group_df, x="experiment_segment", y="rate", hue="type",palette=my_palette,
    dodge=True, alpha=.4, legend=False,
)
sns.pointplot(group_df, x="experiment_segment", y="rate", hue="type", palette=my_palette,
              errorbar=None, linestyle="none",
              estimator="median",   marker="_", markersize=20, markeredgewidth=3, dodge=.4)
plt.show()

