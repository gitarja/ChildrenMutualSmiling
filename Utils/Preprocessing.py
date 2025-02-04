import numpy as np



def avgSliding(data:np.array, window_n:int= 3):
    '''
    :param data: sequence of data
    :param window_n: the lenght of the average sliding window
    :return:
    '''

    return np.convolve(data, np.ones(window_n) / window_n, mode="valid")










