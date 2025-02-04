import numpy as np
import pandas as pd
import os
from Utils.Conf import FPS
data_path = "F:\\users\\prasetia\\data\\Children\\children_sync\\data\\smile_detection\\"

file_name = "Group_27_D_poster_final.pkl"
new_events = [
[2, 9],
[2, 59],

              ]


df = pd.read_pickle(os.path.join(data_path, file_name))




for ne in new_events:
    minute = ne[0] * (FPS * 60)
    second = ne[1] * FPS

    i_frame_start = minute + second

    i_frame_end = i_frame_start+ (FPS * 2)

    df2 = {'start': i_frame_start, 'stop': i_frame_end, 'label': 1, 'index': -1, 'status': 1}
    df = df._append(df2, ignore_index=True)


print(df)
df["index"] = np.arange(len(df))
pd.to_pickle(df, os.path.join(data_path, "append", file_name))