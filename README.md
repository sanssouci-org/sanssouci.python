# sanssouci.python

Post hoc inference via multiple testing

The goal of this projet is to port the [R package sansSouci](pneuvial.github.io/sanssouci/) to python.

Authors: Laurent Risser and Pierre Neuvial


## Test the package on synthetic data


```
import sanssouci as sa
import numpy as np


#1) generate phantom data
p = 130
n = 45


X=np.random.randn(n,p)  #NOTE: no signal!! we expect trivial bounds
categ=np.random.binomial(1, 0.4, size=n)


#2) test the algorithm
B = 100
pval0=sa.get_perm_p(X, categ, B=B , row_test_fun=sa.row_welch_tests)

K=p
piv_stat=sa.get_pivotal_stats(pval0, K=K)


#3) Compute Bounds

alpha=0.1

lambda_quant=np.quantile(piv_stat, alpha)
thr=sa.t_linear(lambda_quant, np.arange(1,p+1), p)
swt=sa.row_welch_tests(X, categ)
p_values=swt['p_value'][:]
pvals=p_values[:10]

bound = sa.max_fp(pvals, thr)
print(bound)

```
