import arviz as az
from cycler import cycler
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
def myStyle():
    sns.set_theme()
    #sns.set(font_scale=5)
    sns.set(font="Arial")
    sns.set_style("whitegrid")
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams.update({'xtick.labelsize': 8, 'ytick.labelsize': 8})
    sns.set_style('ticks')


    plt.rcParams["axes.prop_cycle"] = cycler(color=['#252525', '#525252', '#737373', '#969696'])