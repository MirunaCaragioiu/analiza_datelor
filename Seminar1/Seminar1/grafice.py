import matplotlib.pyplot as plt
import seaborn as sb

def corelograma(t,vmin=-1,vmax=1):
    sb.heatmap(t,vmin,vmax,"RdYlBu")
    plt.show()
