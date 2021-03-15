import pandas as pd
import functii
import sklearn.cross_decomposition as cdec
import numpy as np
import grafice
import sklearn.preprocessing as pp

t_emisii = pd.read_csv("EnergieSiMediu/EnergieSiMediu/Emissions.csv", index_col=0)
t_electricitate = pd.read_csv("EnergieSiMediu/EnergieSiMediu/ElectricityProduction.csv", index_col=0)

var1 = list(t_emisii)[1:]
var2 = list(t_electricitate)[1:]

t = t_emisii.join(other=t_electricitate, how="inner", lsuffix="_1", rsuffix="_2")

x = t[var1].values
y = t[var2].values

# print("x:",x,"y:",y,sep="\n")
functii.inlocuire_nan(x)
functii.inlocuire_nan(y)
# Construire model analiza canonica
n, p = x.shape
q = y.shape[1]
m = min(p, q)
model_ac = cdec.CCA(n_components=m, scale=False)
model_ac.fit(x, y)
# Preluare rezultate
# Scoruri
z = model_ac.x_scores_
u = model_ac.y_scores_
# Normalizare scoruri
pp.normalize(z, axis=0, copy=False)
pp.normalize(u, axis=0, copy=False)
print("z:", z, "u:", u, sep="\n")
r = np.diagonal(np.corrcoef(z, u, rowvar=False)[:m, m:])
print("Corelatii canonice:", r)
p_values = functii.test_bartlett_wilks(r, n, p, q, m)
print("Test Bartlett. P-Values:", p_values)

ryu = np.corrcoef(y, u[:, :2], rowvar=False)[:q, q:]
rxz = np.corrcoef(x, z[:, :2], rowvar=False)[:p, p:]
print("Corelatii variabile-variabile canonice:")
print(rxz, ryu, sep="\n")
grafice.plot_corelatii(rxz[:, 0], rxz[:, 1],
                       ryu[:, 0], ryu[:, 1],
                       var1, var2)
