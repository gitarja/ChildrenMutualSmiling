import matplotlib.pyplot as plt
import numpy as np
from Utils.Preprocessing import avgSliding
from sklearn.cluster import KMeans
from Utils.Plotting import drawAxis
from Utils.SpatioTemporal import similarityScore

from sklearn.metrics import silhouette_score
results_path = "F:\\users\\prasetia\\data\\Children\\children_sync\\results\\bc\\"

head_pose = np.load(results_path + "head_pos.npy")

head_rot = np.load(results_path + "head_ort.npy")


# setting up paramas
sliding_window_n = 5
duration = 60
fps = 30
# reshape the array
N_row, _, _ = head_pose.shape

head_pose = head_pose.reshape((N_row, -1)) # (x_1, y_1, x_2, y_2, ..)
head_rot = head_rot.reshape((N_row, -1)) # (pitch_1, yaw_1, roll_1, pitch_2, ..)

# normalization

norm_head_pose = np.array([avgSliding(data=head_pose[:, i], window_n=sliding_window_n) for i in range(head_pose.shape[-1])]).transpose()
norm_head_rot = np.array([avgSliding(data=head_rot[:, i], window_n=sliding_window_n) for i in range(head_rot.shape[-1])]).transpose()

N, M = norm_head_rot.shape
N_sub = int(M / 3)
a = norm_head_rot.reshape((N * N_sub, 3))

kmeans = KMeans(n_clusters=4, random_state=0).fit(a)
# drawAxis(kmeans.cluster_centers_)

head_rot_idx = kmeans.labels_.reshape(N, N_sub)
head_rot_idx = head_rot_idx[:, [4, 1, 0, 2, 3, 5]]
adj_mat = np.zeros((N_sub, N_sub, 11))

for i in range(N_sub-1):

    for j in range(i+1, N_sub, 1):
        adj_mat[i, j]= similarityScore(head_rot_idx[:, i], head_rot_idx[:, j], duration=duration, fps=fps)
        adj_mat[j, i] = similarityScore(head_rot_idx[:, i], head_rot_idx[:, j], duration=duration, fps=fps)
        # print(adj_mat)



np.save(results_path + "adj_mat.npy", adj_mat)