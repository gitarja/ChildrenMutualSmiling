import matplotlib.pyplot as plt
import numpy as np
from Utils.Visualization import plotGraph
from Utils.DataReader import DataReader
from Utils.SyncMetrics import computeAdjacencyMetrix
import pandas as pd


animation_path = "F:\\users\\prasetia\\projects\\Animations\\ChildrenSync\\"
results_path = "F:\\users\\prasetia\\data\\Children\\children_sync\\data\\"
reader = DataReader(results_path=results_path)

for i in range(1, 34, 1):

    streams, story_idx, discussion_idx = reader.getData(group_num=i)


    # streams = np.asarray(streams)






