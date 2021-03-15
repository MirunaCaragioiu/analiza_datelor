import pandas as pd
import sklearn.discriminant_analysis as disc
import grafice
import sklearn.metrics as metrics
import numpy as np
import sklearn.naive_bayes as nb


set_invatare = pd.read_csv("ParkinsonsDataSet/park.csv",index_col=0)
variabile = list(set_invatare)
nr_var = len(variabile)
variabile_predictor = variabile[:(nr_var-1)]
variabila_tinta = variabile[nr_var-1]

x = set_invatare[variabile_predictor].values
y = set_invatare[variabila_tinta].values

# print("x:",x,"y:",y,sep="\n")
# Creare model liniar
model_ALD = disc.LinearDiscriminantAnalysis(solver="eigen")
model_ALD.fit(x,y)
# Preluare rezultate si aplicare model
# Calcul scoruri discriminante (functii Fisher)
z = model_ALD.transform(x)
clase = model_ALD.classes_
grafice.distributie(z,0,y,clase)
# Preluare si afisare functii de clasificare
F = model_ALD.coef_
F0 = model_ALD.intercept_
print("Functii clsificare:",F,F0)
# Aplicare model pe setul de invatare
clasificare = model_ALD.predict(x)
tabel_clasificare = pd.DataFrame({variabila_tinta:y,"Predictia":clasificare},
                                 index=set_invatare.index)
tabel_clasificare.to_csv("ClasificareSetInvatare.csv")
tabel_clasificare_err = tabel_clasificare[y!=clasificare]
tabel_clasificare_err.to_csv("ClasificariEronate.csv")
# Clacul acuratete clasificare
acuratete_globala = metrics.accuracy_score(y,clasificare)
print("Acuratete globala:",acuratete_globala)
# Calcul matrice de confuzie
mat_conf = metrics.confusion_matrix(y,clasificare)
tabel_confuzie = pd.DataFrame(data=mat_conf,index=clase,columns=clase)
tabel_confuzie["Acuratete"] = np.diagonal(mat_conf)*100/np.sum(mat_conf,axis=1)
print(tabel_confuzie)
print("Coeficient Cohen-Kappa:",metrics.cohen_kappa_score(y,clasificare))
# Aplicarea modelului pe setul de test

set_aplicare = pd.read_csv("ParkinsonsDataSet/park_test.csv",index_col=0)
x_ = set_aplicare[variabile_predictor].values

clasificare_test = model_ALD.predict(x_)
set_aplicare["Predictia"] = clasificare_test
set_aplicare.to_csv("park_test_clasificare.csv")

# Creare si aplicare model Bayesian
model_Bayes = nb.GaussianNB()
model_Bayes.fit(x,y)
clasificare_nb = model_Bayes.predict(x)
mat_conf_nb = metrics.confusion_matrix(y,clasificare_nb)
tabel_confuzie_nb = pd.DataFrame(mat_conf_nb,clase,clase)
tabel_confuzie_nb["Acuratete"] = np.diagonal(mat_conf_nb)*100/np.sum(mat_conf_nb,axis=1)
print('Tabel confuzie model bayesian:',tabel_confuzie_nb,sep='\n')




