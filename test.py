import scipy
import numpy as np
from sanssouci import sanssouci as s2
import matplotlib.pyplot as plt


#1) generate phantom data
p = 130
n = 45


X=np.random.randn(n,p)  ## NOTE: no signal !! we expect trivial bounds
categ=np.random.binomial(1, 0.4, size=n)


#2) test the algorithm
B = 100
pval0=s2.get_perm_p(X, categ, B=B , row_test_fun=s2.row_welch_tests)

K=p
piv_stat=s2.get_pivotal_stats(pval0, K=K)


#3) Compute Bounds

alpha=0.1
  
lambda_quant=np.quantile(piv_stat, alpha)
thr=s2.t_linear(lambda_quant, np.arange(1,p+1), p)
swt=s2.row_welch_tests(X, categ)
p_values=swt['p_value'][:]
pvals=p_values[:10]  

bound = s2.max_fp(pvals, thr)
print(bound)
