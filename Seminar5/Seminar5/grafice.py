import matplotlib.pyplot as plt
import seaborn as sb


def distributie(z, k, y, clase):
    fig = plt.figure(figsize=(10, 8))
    axe = fig.add_subplot(1, 1, 1)
    assert isinstance(axe, plt.Axes)
    axe.set_title("Distributie in axa discriminanta " + str(k), fontsize=16, color='b')
    for g in clase:
        sb.kdeplot(z[y == g, k], shade=True, ax=axe,label = g)
    plt.show()
