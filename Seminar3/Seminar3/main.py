import pandas as pd
import sklearn.decomposition as dec
import functii
import numpy as np
import factor_analyzer as fact

t = pd.read_csv("Freelancer/Freelancer/FreeLancerT.csv", index_col=1)
variabile = list(t)
variabile_prelucrate = variabile[2:]
# print(variabile)
x = t[variabile_prelucrate].values
n, m = np.shape(x)
# print(x)
functii.inlocuire_nan(x)
# print(x)
# Construire model PCA - Analiza in componente principale
model_pca = dec.PCA()
x_std = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
model_pca.fit(x_std)
# Calcul scoruri
s = model_pca.transform(x_std)
print("Componente (scoruri) ACP sklearn:", s, sep="\n")
functii.plot_scoruri(s[:, 0], s[:, 1], list(t.index), "C1", "C2", "Plot scoruri - ACP")
rxs_ = np.corrcoef(x, s, rowvar=False)
rxs = rxs_[:m, m:]
print("Corelatii variabile - componente ACP sklearn:", rxs, sep="\n")
alpha = model_pca.explained_variance_
a = model_pca.components_

# Analiza factoriala
model_fa = fact.FactorAnalyzer(rotation=None)
model_fa.fit(x)
# Calcul scoruri
f = model_fa.transform(x)
print("Factori comuni (factor_analyzer):", f, sep="\n")
functii.plot_scoruri(f[:, 0], f[:, 1], list(t.index), "F1", "F2", "Plot scoruri - Analiza Factoriala")
l = model_fa.loadings_
print("Legaturi variabile - factori comuni (factor_analyzer):", l, sep="\n")
alpha_fa = model_fa.get_factor_variance()
print("Varianta explicata de factori (factor_analyzer):",alpha_fa,sep="\n")
# Model factorial sklearn
model_fa_sk = dec.FactorAnalysis(n_components=3)
model_fa_sk.fit(x)
f_sk = model_fa_sk.transform(x)
print("Factori comuni (sklearn):", f_sk, sep="\n")
functii.plot_scoruri(f_sk[:, 0], f_sk[:, 1], list(t.index), "F1", "F2", "Plot scoruri - Analiza Factoriala - sk")
functii.show()
