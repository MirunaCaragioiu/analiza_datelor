import matplotlib.pyplot as plt
import numpy as np


def plot_corelatii(rxz1, rxz2, ryu1, ryu2, var1, var2):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.plot(np.cos(np.arange(0,2*np.pi,0.01)),np.sin(np.arange(0,2*np.pi,0.01)),color='k')
    ax.axhline(0);ax.axvline(0)
    ax.scatter(rxz1, rxz2, c='r', label="X-Z")
    ax.scatter(ryu1, ryu2, c='b', label="Y-U")
    p = len(var1)
    for i in range(p):
        ax.text(rxz1[i], rxz2[i], var1[i])
    q = len(var2)
    for i in range(q):
        ax.text(ryu1[i], ryu2[i], var2[i])
    ax.legend()
    plt.show()
