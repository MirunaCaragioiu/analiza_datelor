import numpy as np
import matplotlib.pyplot as plt

def inlocuire_nan(x):
    isnan = np.isnan(x)
    assert isinstance(isnan,np.ndarray)
    if isnan.any():
        k = np.where(isnan)
        x[k] = np.nanmean(x[:,k[1]],axis=0)

def plot_scoruri(x,y,label,lx,ly,title):
    fig = plt.figure(figsize=(10,7))
    assert isinstance(fig,plt.Figure)
    ax = fig.add_subplot(1,1,1)
    assert isinstance(ax,plt.Axes)
    ax.set_title(title,fontsize = 16, color = 'b')
    ax.set_xlabel(lx)
    ax.set_ylabel(ly)
    ax.scatter(x,y,c='r')
    n = len(label)
    for i in range(n):
        ax.text(x[i],y[i],label[i])
    # plt.show()

def show():
    plt.show()