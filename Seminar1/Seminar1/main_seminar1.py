import pandas as pd
import calcule
import grafice

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
