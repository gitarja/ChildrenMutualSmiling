import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from dtaidistance import dtw

def similarityScore(x1, x2, duration=60, fps=30):

    window_n_length = duration * fps
    sim_scores = np.array([dtw.distance(x1[i:i+window_n_length], x2[i:i+window_n_length]) for i in range(0, len(x1), window_n_length)])


    return sim_scores
