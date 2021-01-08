import numpy as np
from scipy import stats

from .row_welch import row_welch_tests
from .reference_families import t_inv_linear

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# LAMBDA-CALIBRATION
# R source code: https://github.com/pneuvial/sanssouci/
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_perm_p(X, categ, B=100 , row_test_fun=stats.ttest_ind):
  """
  Get permutation p-values: Get a matrix of p-values under the null
  hypothesis obtained by repeated permutation of class labels.

  Inputs:
  - X: numpy array of size [n,p], containing n observations of p variables (hypotheses)
  - categ: numpy array of size [n], containing n values in {0, 1}, each of them specifying
           the column indices of the first and the second sample.
  - B: number of permutations to be performed (default=100)
  - row_test_fun: testing function with the same I/O as 'stats.ttest_ind' (default).
                Specifically, must have two lists as inputs (or 1d np.arrays) for the
                compared data, and the resulting pvalue must be accessed in '[test].pvalue'
                Eligible functions are for instance "stats.ks_2samp",
                "stats.bartlett", "stats.ranksums", "stats.kruskal"

  Returns:
  - pval0:  A numpy array of size [B,p], whose entry i,j corresponds to p_{(j)}(g_i.X)
            with notation of the AoS 2020 paper cited below (section 4.5)

  Reference:
  - Blanchard, G., Neuvial, P., & Roquain, E. (2020). Post hoc confidence bounds on false
     positives using reference families. Annals of Statistics, 48(3), 1281-1303.
  """

  #Init
  n=X.shape[0]
  p=X.shape[1]

  #Step 1: calculate $p$-values for B permutations of the class assignments

  #1.1: Intialise all vectors and matrices
  shuffled_categ_current=categ.copy()

  shuffled_categ_all=np.zeros([B,n])
  for bb in range(B):
    np.random.shuffle(shuffled_categ_current)
    shuffled_categ_all[bb,:]=shuffled_categ_current[:]

  #1.2: calculate the p-values
  pval0=np.zeros([B,p])

  if row_test_fun==row_welch_tests: #row Welch Tests (parallelized)
    for bb in range(B):
      swt=row_welch_tests(X, shuffled_categ_all[bb,:])
      pval0[bb,:]=swt['p_value'][:]
  else:                     #standard scipy tests
    for bb in range(B):
      s0 = np.where(shuffled_categ_all[bb,:]==0)[0]
      s1 = np.where(shuffled_categ_all[bb,:]==1)[0]

      for ii in range(p):
        rwt=row_test_fun(X[s0, ii], X[s1, ii])  #Welch test with scipy -> rwt=stats.ttest_ind(X[s0, ii], X[s1, ii],equal_var=False)

        pval0[bb,ii] = rwt.pvalue

  # Step 2: sort each column
  pval0 = np.sort(pval0,axis=1)

  return pval0



def get_pivotal_stats(p0,  t_inv=t_inv_linear, K = -1):
  """
  Get pivotal statistic

  Inputs:
  - p0:  A numpy array of size [B,p] of null p-values obtained from B permutations
         for p hypotheses.
  - t_inv: A function with the same I/O as t_inv_linear
  -  K:  For JER control over 1:K, i.e. joint control of all k-FWER, k<= K.
         Automatically set to p if its input value is < 0.

  Returns:
   - A numpy array of of size [B]  containing the pivotal statitics, whose j-th entry
     corresponds to \psi(g_j.X) with notation of the AoS 2020 paper cited below (section 4.5)

   Reference:
   - Blanchard, G., Neuvial, P., & Roquain, E. (2020). Post hoc confidence bounds on false
     positives using reference families. Annals of Statistics, 48(3), 1281-1303.
  """

  # Step 3: apply template function
  tkInv_all=t_inv(p0)


  if K<0:
    K=tkInv_all.shape[1]  # tkInv_all.shape[1] is equal to p

  # Step 4: report min for each row
  piv_stat=np.min(tkInv_all[:,:K],axis=1)

  return piv_stat
