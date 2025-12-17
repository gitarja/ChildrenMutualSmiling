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

def generateSurrogate(stream):
    def get_one_episodes(arr):
        """
        Extract episodes of consecutive 1s from a 1D array of 0/1 values.
        Returns an array of shape (N, 2) where:
            - col 0 = start index of the run of 1s
            - col 1 = end index of the run of 1s (inclusive)
        """
        arr = np.asarray(arr)

        # Detect rising edges (0 -> 1) and falling edges (1 -> 0)
        diff = np.diff(arr, prepend=0, append=0)

        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1

        return np.vstack([starts, ends]).T

    a = get_one_episodes(stream)
    duration = (a[:, 1] - a[:, 0])
    np.random.permutation(duration)
    n = len(stream)

    new_start_idx = np.random.randint(0, n-np.max(duration), size=len(a))

    surrogate_stream = np.zeros_like(stream)
    for i in range(len(duration)):
        surrogate_stream[new_start_idx[i]:new_start_idx[i]+duration[i]+1] = 1

    return surrogate_stream

def dyadicOverlap(a, b, lags=[0, 30, 60, 120, 240, 360], max=False):

    def computeSync(a, b, lag):
        a_idx = groupPoints(np.nonzero(a == 1)[0], within_th=(lag)//2)
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

    sync_score = np.zeros(shape=(len(lags)))
    i=0
    for l in lags:
        sync_score[i] = (computeSync(a, b, l))
        i+=1

    if max:
        return np.max(sync_score)
    else:

        return sync_score, lags


def crossCorrelation(a, b):
    tau_max = 360

    value_positive = []
    value_negative = []

    temp_1 = (a - np.mean(a)) / np.std(a)
    temp_2 = (b - np.mean(b)) / np.std(b)
    # temp_1 = a
    # temp_2 = b

    value_positive.append(np.mean(temp_1 * temp_2))

    for tau in range(0, len(temp_1) // 2, 60):
        in_data = np.pad(a[tau:], (0, tau), "constant")
        out_data = np.pad(b[tau:], (0, tau), "constant")

        # move out_data
        value_positive.append(np.mean(temp_1 * ((out_data - np.mean(out_data)) / np.std(out_data))))
        # move in_data
        value_negative.append(np.mean(temp_2 * ((in_data - np.mean(in_data)) / np.std(in_data))))

    value_series = np.array(value_positive + value_negative)
    return value_series


def computeAdjacencyMetrix(streams, lag=60, against_surrogate=False):

    adj_matrix = np.zeros((len(streams), len(streams)))
    for i in range(len(streams)):
        a = streams[i]
        for j in range(i+1, len(streams), 1):
            b = streams[j]
            if against_surrogate:
                b = generateSurrogate(b, groupPoints(np.nonzero(b == 1)[0], within_th=(lag) // 2))
            adj_matrix[i, j]=  dyadicOverlap(a, b, max=True)
            crossCorrelation(a, b)
            adj_matrix[j, i] = dyadicOverlap(b, a, max=True)


    return adj_matrix