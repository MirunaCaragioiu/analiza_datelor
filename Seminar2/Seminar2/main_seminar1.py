import pandas as pd
import calcule
import grafice
import numpy as np

nume_fisier = "ADN/ADN/ADN_Total.csv"
# Citire din fisier in DataFrame
tabel = pd.read_csv(nume_fisier,index_col=0)
# print(tabel)
# Preluare nume de variabile in obiect list
variabile = list(tabel)
instante = list(tabel.index)
# print(variabile,instante,sep="\n")
x = tabel[variabile].values
# print(x,type(x),sep="\n")
r, alpha, a, c = calcule.acp(x)
t_r = pd.DataFrame(r,variabile,variabile)
t_r.to_csv("R.csv")
grafice.corelograma(t_r)
t_varianta = calcule.tabelare_valori_proprii(alpha)
t_varianta.to_csv("Varianta.csv")
# Calcul corelatii factoriale
rxc = a*np.sqrt(alpha)
# Tabelare corelatii factoriale
nume_componente = ["Comp"+str(i) for i in range(1,len(alpha)+1)]
t_rxc = pd.DataFrame(data=rxc,index=variabile,columns=nume_componente)
t_rxc.to_csv("rxc.csv")
# Vizualizare corelograma corelatii factoriale
grafice.corelograma(t_rxc,titlu="Corelograma corelatii factoriale")
# Calcul scoruri
s = c/np.sqrt(alpha)
t_s = pd.DataFrame(data=s,index=instante,columns=nume_componente)
t_s.to_csv("s.csv")
# Plot instante
grafice.scatter_plot(s,nume_instante=instante,titlu="Plot instante dupa scoruri")
# Calcul cosinusuri
c2 = c * c
cos = np.transpose(np.transpose(c2) / np.sum(c2,axis=1))
t_cos = pd.DataFrame(data=cos,index=instante,columns=nume_componente)
t_cos.to_csv("cos.csv")
# Calcul contributii
n = len(instante)
beta = c2/(n*alpha)
t_beta = pd.DataFrame(data=beta,index=instante,columns=nume_componente)
t_beta.to_csv("contributii.csv")
# Calcul comunalitati
comm = np.cumsum(rxc*rxc,axis=1)
t_comm = pd.DataFrame(data=comm,index=variabile,columns=nume_componente)
t_comm.to_csv("comunalitati.csv")
# Corelograma comunalitati
grafice.corelograma(t_comm,vmin=0,titlu="Corelograma comunalitati")
grafice.plt_show()
