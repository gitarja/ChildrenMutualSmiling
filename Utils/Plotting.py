import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin



def drawAxis(data, tdx=10, tdy=10, size = 50):


    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy

    fig, axs = plt.subplots(2, 2)
    i, j = 0, 0
    for d in data:


        pitch = d[0] * np.pi / 180
        yaw = -(d[1] * np.pi / 180)
        roll = d[2] * np.pi / 180

        # X-Axis pointing to right. drawn in red
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy


        axs[i, j].plot([tdx, x1], [tdy, y1], color="red")
        axs[i, j].plot([tdx, x2], [tdy, y2], color="green")
        axs[i, j].plot([tdx, x3], [tdy, y3], color="blue")

        axs[i, j].invert_yaxis()


        if j == 1:
            j=0
            i+=1
        else:
            j += 1


    plt.show()


