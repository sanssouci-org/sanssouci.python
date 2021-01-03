import scipy
import numpy as np
import sanssouci.tests as tst
import sanssouci.reference_families as fam
import sanssouci.calibration as cal
import sanssouci.posthoc_bounds as bounds


#1) generate phantom data
p = 130
n = 45


"""
X_df=pd.read_csv("CodeR_mini/X.csv")
X=X_df.values[:,1:].transpose()

categ_df=pd.read_csv("CodeR_mini/categ.csv")
categ=categ_df.values[:,1]
print("X.csv and categ.csv downloaded")
"""

X=np.random.randn(n,p)
categ=np.random.binomial(1, 0.4, size=n)


#2) test the algorithm
B = 100
pval0=cal.get_perm_p(X, categ, B=B , row_test_fun=tst.row_welch_tests)

K=p
piv_stat=cal.get_pivotal_stats(pval0, K=K)


#3) Compute Bounds

alpha=0.1
  
lambda_quant=np.quantile(piv_stat, alpha)
thr=t_linear(lambda_quant, np.arange(1,p+1), p)
swt=tst.row_welch_tests(X, categ)
p_values=swt['p_value'][:]
pvals=p_values[:10]  

bound = bounds.max_fp(pvals, thr)

bounds=bounds.curve_max_fp(pvals, thr)
print(bounds)
