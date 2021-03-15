import matplotlib.pyplot as plt
import seaborn as sb

def corelograma(t,vmin=-1,vmax=1,titlu= "Corelograma"):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(1,1,1)
    ax.set_title(titlu,fontsize=16)
    sb.heatmap(t,vmin,vmax,"RdYlBu",ax=ax)

# Plot instante
def scatter_plot(x,k1=0,k2=1,nume_instante=None,titlu="Plot instante"):
    fig = plt.figure(figsize=(10,7))
    assert isinstance(fig,plt.Figure)
    ax = fig.add_subplot(1,1,1)
    assert isinstance(ax,plt.Axes)
    ax.set_title(titlu,fontsize=16)
    ax.set_xlabel("a"+str(k1+1))
    ax.set_ylabel("a"+str(k2+1))
    ax.scatter(x[:,k1],x[:,k2],c = 'b')
    if nume_instante is not None:
        n = len(nume_instante)
        for i in range(n):
            ax.text(x[i,k1],x[i,k2],nume_instante[i])

def plt_show():
    plt.show()
