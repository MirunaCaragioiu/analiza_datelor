import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as hclust
import grafice
import sklearn.cluster.hierarchical as skhclust

mortalitate = pd.read_csv("MortalitateEU/MortalitateEU.csv", index_col=0)
variabile = list(mortalitate)[1:]
nume_instante = list(mortalitate.index)

x = mortalitate[variabile].values
if np.isnan(x).any():
    print("Valori lipsa!!!")
    k = np.where(np.isnan(x))
    x[k] = np.nanmean(x[:, k[1]], axis=0)
# print(variabile, x, sep="\n")
# Creare ierarhie prin scipy
metoda = "ward"
h = hclust.linkage(x, method=metoda)
print("Matrice ierarhie:", h, sep="\n")
# Plot ierarhie - garficul dendrograma
grafice.dendrograma(
    h,
    nume_instante,
    "Plot ierarhie. Metoda:" + metoda +
    " Metrica euclidiana"
)
# Determinare partitie optimala
m = np.shape(h)[0]
# Numar clusteri din partitia optimala:
k_opt = m - np.argmax(h[1:, 2] - h[:(m - 1), 2])
print("Partitia optimala are " + str(k_opt) + " clusteri")

# Construire model clusterizare sklearn
model_clusterizare_sk = skhclust.AgglomerativeClustering(n_clusters=k_opt, linkage=metoda)
model_clusterizare_sk.fit(x)
coduri = model_clusterizare_sk.labels_
partitie = np.array(["Cluster" + str(cod + 1) for cod in coduri])
tabel_partitie = pd.DataFrame(
    data={"Partitie": partitie},
    index=mortalitate.index
)
print("Partitia optimala:", tabel_partitie, sep="\n")

# Plot partitie in axele principale
grafice.plot_partitie(
    x,
    partitie,
    nume_instante,
    "Plot partitie optimala"
)
grafice.show()
