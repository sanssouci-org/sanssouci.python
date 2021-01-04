import numpy as np
from scipy import stats

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROW WELSH TESTS
# R source code: https://github.com/pneuvial/sanssouci/
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def get_summary_stats(mat, categ):
  """
  Convert a matrix of observations labelled into categories into summary 
  statistics for each category
  
  The following statistics are calculated: sums, sums of squares, means, 
  standard deviations, sample sizes
   
  * Inputs:
    - mat: Numpy array matrix whose columns correspond to the p variables and 
    rows to the n observations
    - categ:  A numpy array of size n representing the category of each 
    observation, in {0, 1} 
  
  * returns:
    - A dictionary containing the above-described summary statistics for each 
      category
  """
  
  cats=set(categ)
  
  res={}
  
  for cc in cats:
    ww = np.where(categ[:]==cc)[0]
    matc=mat[ww, :]
    sumc=np.sum(matc,axis=0)
    sum2c=np.sum(matc*matc,axis=0)
    nc=ww.shape[0]
    mc=sumc/nc
    sc=np.sqrt((sum2c-((sumc*sumc)/nc))/(nc-1))
    
    res[cc] = {"sum":sumc, "sum2":sum2c, "n":nc, "mean":mc, "sd":sc}
    
  return res


def suff_welch_test(mx, my, sx, sy, nx, ny):
  """
  Welch test from sufficient statistics
  
  * Inputs:
    - mx: A numeric value or vector, the sample average for condition "x"
    - my A numeric value or vector of the same length as 'mx', the sample
      average for condition "y"
    - sx A numeric value or vector of the same length as 'mx', the standard
      deviation for condition "x"
    - sy A numeric value or vector of the same length as 'mx', the standard
      deviation for condition "y"
    - nx A numeric value or vector of the same length as 'mx', the sample
      size for condition "x"
    - ny A numeric value or vector of the same length as 'mx', the sample
      size for condition "y"
  
    Note that the alternative hypothesis is "two.sided". It could be extended
    to "greater" or "less" as in the original R code.
  
  * Returns: A dictionary with elements:
    - statistic: the value of the t-statistic
    - parameter:  the degrees of freedom for the t-statistic
    - p_value: the p-value for the test
  """
  
  #pre-computations
  sse_x=(sx*sx)/nx
  sse_y=(sy*sy)/ny
  sse=sse_x+sse_y
  sse2=sse*sse
  
  #test statistic
  stat=(mx-my)/np.sqrt(sse)
  
  #approximate degrees of freedom (Welch-Satterthwaite)
  df=sse2 / ( (sse_x*sse_x/(nx-1)) + ((sse_y*sse_y)/(ny-1)))
  
  #p-value
  pval= 2*(1 - stats.t.cdf(np.abs(stat),df=df))
  
  return {"statistic":stat , "parameter":df , "p_value":pval}


def row_welch_tests(mat, categ):
  """
  Welch t-tests for each column of a matrix, intended to be speed efficient
  
  Note that the alternative hypothesis is "two.sided". It could be extended
  to "greater" or "less" as in the original R code.
  
  * Inputs:
    - mat: Numpy array matrix whose columns correspond to the p variables and 
    rows to the n observations
    - categ:  A numpy array of size n representing the category of each 
    observation, in {0, 1} 
    
  * Returns: ...
  
  * References: B. L. Welch (1951), On the comparison of several mean values: an
                alternative approach. Biometrika, 38, 330-336

  """
  
  n=mat.shape[0]
  p=mat.shape[1]
  
  sstats=get_summary_stats(mat, categ)
  
  Y=sstats[0]
  X=sstats[1]


  swt=suff_welch_test(X["mean"], Y["mean"],X["sd"], Y["sd"],X["n"], Y["n"])
  swt["meanDiff"]=X["mean"]-Y["mean"]
  
  return swt

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# REFERENCE FAMILIES
# R source code: https://github.com/pneuvial/sanssouci/
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def  t_inv_linear(p0):
  """
  Warning: In case the last command does not work, you can use alternatively:
  
  out=pval0_sorted_all*1.
  for i in range(pval0_sorted_all.shape[0]):
    out[i,:]=pval0_sorted_all[i,:]/normalized_ranks[:]
  return out
  """
  
  p=p0.shape[1]
  
  normalized_ranks = (np.arange(p)+1)/float(p)
  
  return p0/normalized_ranks



def  t_linear(alpha, k, m):
  return alpha * k / (m*1.0)

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
  
  #BEGIN HACK
  #pd_shuffled_categ_all=pd.read_csv("CodeR_mini/all_categ_R.csv")
  #shuffled_categ_all[:,:]=pd_shuffled_categ_all.values[:,1:]
  #print("CodeR_mini/all_categ_R.csv was loaded")
  #END HACK

  
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

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# POST HOC BOUNDS
# R source code: https://github.com/pneuvial/sanssouci/
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def max_fp(p_values, thr):
  """
  Upper bound for the number of false discoveries in a selection
  
  * Inputs:
    - p_values: A 1D numpy array of p-values for the selected items
    - thr: A 1D numpy array of non-decreasing k-FWER-controlling thresholds
  * Returns:
    - A post hoc upper bound on the number of false discoveries in the selection
  * Reference:
    - Blanchard, G., Neuvial, P., & Roquain, E. (2020). Post hoc confidence bounds 
    on false positives using reference families. Annals of Statistics, 48(3), 1281-1303.
  """
  
  #make sure that thr is sorted
  if np.linalg.norm(thr - thr[np.argsort(thr)])>0.0001:
    thr=thr[np.argsort(thr)]
    print("The input thr was not sorted -> this is done now")

  #do the job
  nS=p_values.shape[0]
  K=thr.shape[0]
  size=np.min([nS,K])
  
  if size<1:
    return 0 

  seqK=np.arange(size)
  
  thr=thr[seqK]  ## k-FWER control for k>nS is useless (will yield bound > nS)
  
  card=np.zeros(thr.shape[0])
  for i in range(thr.shape[0]):
    card[i]=np.sum(p_values > thr[i])   #card<-sapply(thr,FUN=function(thr){sum(p_values > thr)})
    

  
  return np.min( [nS , (card + seqK).min()])




def min_tp(p_values, thr):
  """
  Lower bound for the number of true discoveries in a selection
  
  * Inputs:
    - p_values: A 1D numpy array of p-values for the selected items
    - thr: A 1D numpy array of non-decreasing k-FWER-controlling thresholds
  * Returns:
    - A Lower bound on the number of true discoveries in the selection
  * Reference:
    - Blanchard, G., Neuvial, P., & Roquain, E. (2020). Post hoc confidence bounds 
    on false positives using reference families. Annals of Statistics, 48(3), 1281-1303.
  """
  
  return p_values.shape[0] - max_fp(p_values, thr)




def curve_max_fp(p_values, thr):
  """
  Upper bound for the number of false discoveries among most significant items

  * Inputs:
    - p_values: A 1D numpy array containing all $p$ p-values, sorted non-decreasingly
    - thr: A 1D numpy array  of $K$ JER-controlling thresholds, sorted non-decreasingly
  * Returns:
    - A vector of size p giving an joint upper confidence bound on  the number of 
      false discoveries among the $k$ most significant items for all k in \{1,\ldots,m\}
  * Reference:
    - Blanchard, G., Neuvial, P., & Roquain, E. (2020). Post hoc confidence bounds 
      on false positives using reference families. Annals of Statistics, 48(3), 1281-1303.
  """

  #make sure that p_values and thr are sorted
  if np.linalg.norm(p_values - p_values[np.argsort(p_values)])>0.0001:
    p_values=p_values[np.argsort(p_values)]
    print("The input p-values were not sorted -> this is done now")
    
  if np.linalg.norm(thr - thr[np.argsort(thr)])>0.0001:
    thr=thr[np.argsort(thr)]
    print("The input thr were not sorted -> this is done now")
  
  #do the job
  p=p_values.shape[0]
  kMax=thr.shape[0]
  
  if kMax < p:
    thr =  np.concatenate((thr,thr[-1]*np.ones(p-kMax)))
    kMax = thr.shape[0]
  
  K =  np.ones(p)*(kMax)  ## K[i] = number of k/ T[i] <= s[k] = BB in 'Mein2006'
  Z = np.ones(kMax)*(p)  ## Z[k] = number of i/ T[i] >  s[k] = cardinal of R_k
  ## 'K' and 'Z' are initialized to their largest possible value, ie 'p' and 'kMax', respectively
  
  kk = 0
  ii = 0
  
  while (kk < kMax) and (ii < p):
    if thr[kk]>=p_values[ii]:
      K[ii]=kk   # doute : K[ii+1]=kk ???
      ii+=1
    else:
      Z[kk]=ii   # doute : Z[kk+1]=ii ???
      kk+=1

  Vbar=np.zeros(p)
  ww=np.where(K>0)[0]
  A = Z - np.arange(0,kMax)
  
  K_ww=K[ww].astype(np.int)
  cummax_A=A.copy()
  for i in range(1,cummax_A.shape[0]):
    cummax_A[i]=np.max([cummax_A[i-1],cummax_A[i]])

  cA = cummax_A[K_ww - 1]  # cA[i] = max_{k<K[i]} A[k]
  
  Vbar[ww] = np.min(np.concatenate(( (ww+1-cA).reshape(1,-1) , (K[ww]).reshape(1,-1) ),axis=0),axis=0)
  
  return Vbar

