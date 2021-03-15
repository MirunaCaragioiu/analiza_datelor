import numpy as np
import pandas as pd


# definire functie pentru analiza in componente principale
def acp(x):
    # assert isinstance(x,np.ndarray)
    n, m = np.shape(x)
    # Calcul matrice de corelatii
    # Standardizare x
    x_std = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    r = (1 / n) * np.transpose(x_std) @ x_std
    # Calcul vectori si valori proprii
    valp, vecp = np.linalg.eig(r)
    k = np.flipud(np.argsort(valp))
    alpha = valp[k]
    a = vecp[:, k]
    # Calcul componente
    c = x_std @ a
    return r, alpha, a, c

def tabelare_valori_proprii(alpha):
    m = len(alpha)
    alpha_cum = np.cumsum(alpha)
    proc = alpha*100/sum(alpha)
    proc_cum = np.cumsum(proc)
    tabel_varianta = pd.DataFrame(data={
        "Varianta":alpha,
        "Varianta cumulata":alpha_cum,
        "Procent varianta":proc,
        "Procent cumulat":proc_cum
    }, index=["Comp"+str(i) for i in range(1,m+1)])
    return tabel_varianta