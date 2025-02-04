import numpy as np
from scipy.ndimage import label#
np.random.seed(1954)
def groupPoints(v: np.array, within_th: int = 5) -> np.array:
    # initialize output
    output = []

    if len(v) > 2:
        # pad array
        v = np.pad(v, 2, mode="edge")
        # label groups of sample that belong to the same peak
        valley_groups, num_groups = label(np.diff(v) < within_th)

        for i in np.unique(valley_groups)[1:]:
            point_group = v[np.where(valley_groups == i)]
            output.append([np.min(point_group), np.max(point_group)])

    return np.array(output)

def generateSurrogate(stream, a: np.array):

    duration = (a[:, 1] - a[:, 0])
    np.random.permutation(duration)
    n = len(stream)

    new_start_idx = np.random.randint(0, n-np.max(duration), size=len(a))

    surrogate_stream = np.zeros_like(stream)
    for i in range(len(duration)):
        surrogate_stream[new_start_idx[i]:new_start_idx[i]+duration[i]+1] = 1

    return surrogate_stream

def dyadicMetrics(streams, lag=60, against_surrogate=False):

    def computeSync(a, b):
        a_idx = groupPoints(np.nonzero(a == 1)[0], within_th=(lag)//2)
        if against_surrogate:
            b = generateSurrogate(b,groupPoints(np.nonzero(b == 1)[0], within_th=(lag)//2))
        b_idx = groupPoints(np.nonzero(b == 1)[0], within_th=(lag)//2)

        if len(a_idx) < len(b_idx):
            ref = a_idx
            pair = b_idx
        else:
            ref = b_idx
            pair = a_idx

        reoccur = 0
        for idx in ref:
            s_idx = idx[0]
            dist_pair = np.abs(s_idx - pair) <= lag
            if np.sum(dist_pair) > 0:
                reoccur +=1

        if len(pair) > 0:
            return reoccur/ len(pair)
        return 0.0

    adj_matrix = np.zeros((len(streams), len(streams)))
    for i in range(len(streams)):
        a = streams[i]
        for j in range(len(streams)):
            b = streams[j]
            adj_matrix[i, j]= computeSync(a, b)


    return adj_matrix