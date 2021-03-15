import numpy as np
import scipy.stats as stt

def inlocuire_nan(x):
    k = np.where(np.isnan(x))
    x[k] = np.nanmean(x[:,k[1]],axis=0)

def test_bartlett_wilks(r,n,p,q,m):
    v = 1-r*r
    chi2 = (-n+1+(p+q+1)/2) * np.log(np.flipud( np.cumprod( np.flipud(v) ) ))
    d = [(p-k+1)*(q-k+1) for k in range(1,m+1)]
    p_values = 1 - stt.chi2.cdf(chi2,d)
    return p_values
