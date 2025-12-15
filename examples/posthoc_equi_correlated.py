# %%
# FDP control on equi-correlatde synthetic data
# =============================================

"""
We observe $n=n_0 + n_1$ independent $p$-dimensional vectors 
$X^{(j)} \sim \mathcal{N}(\mu^{(0)},\Sigma)$ for $1\leq j \leq n_0$, and  
$X^{(j)} \sim \mathcal{N}(\mu^{(1)},\Sigma)$ for $1\leq j \leq n_1$.
Our goal is to test for each feature $i = 1 \dots p$ the null hypothesis:
$\mathcal{H}_{0,i}: \theta^{(0)}_i=\theta^{(1)}_i$. 

We denote by $p_0$ the number of true null hypotheses (that is, the number of
noise features) and by $\pi_0 = p_0/p$ the fraction of true nulls (that is,
the proportion of noise in the data). 

We assume that:

- $\Sigma = \rho + (1-\rho) I_p$, meaning that the correlation between any two
variables is $\rho$; the case $\rho=0$ corresponds to independent tests. 

- for $i = 1, \cdots, p_0$, $\mu^{(0)} = \mu^{(1)} = 0$
- for $i = p_0+1, \cdots, p$, $\mu^{(0)} = 0$ and $\mu^{(1)} = s$,
    where $s$ controls the signal to noise ratio of the problem.

For each feature, the null hypothesis: 
$\mathcal{H}_{0,i}: \theta^{(0)}_i=\theta^{(1)}_i$ is tested by a two-sample
Welch test. 
"""
# %%
# Data simulation
# ---------------

import numpy as np

p = 50000         # number of features     (eg : number of voxels in an image)
n = 80            # number of observations  (eg : number of images)
pi0 = 0.99        # proportion of noise (true null hypotheses)
p0 = int(np.round(pi0 * p))  # number of non active voxels (null hypotheses)
rho = 0.3         # equi-correlation
s = 1             # signal to noise ratio

# generate noise
rng = np.random.default_rng(2025)
Z = rng.normal(size=(n, p))
w = rng.normal(size=(n))
W = np.repeat(w[:, np.newaxis], p, 1)
X = np.sqrt(1 - rho) * Z + np.sqrt(rho) * W

# add signal (for false null hypotheses)
y = np.random.binomial(1, 0.5, size=n)  # two balanced categories
X[y == 1, (p0 + 1): p] += s

# %%
# calculate the p-values
import sanssouci as sa
pval = sa.row_welch_tests(X, y)['p_value']

# %%
# plot the p-values 
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 4))
ax = plt.subplot(1, 2, 1)
ax.hist(pval, 100)  # 100 bins are used for the histogram
ax.set_title('histogram of p-values')

sorted_pval = np.sort(pval)
ax = plt.subplot(1, 2, 2)
ax.plot(sorted_pval[:1000], '.')
ax.set_title('Smallest 1000 p-values')


# %%
# 2. Classical multiple testing: FDR control
# ------------------------------------------

# %%
# A standard way to account for the fact that we are performing many tests
# simultaneously is to control the False Discovery Rate (FDR). The FDR of a
# subset of features is the expected proportion of false positives in this set.
# This is classically done using the Benjamini-Hochberg (1995) procedure.

# The Benjamini-Hochberg (BH) procedure at level $\alpha$ rejects the smallest 
# $index$ $p$-values, where $index$ is the largest crossing point between the 
# sorted $p$-values and the line passing by $(0,0)$ and $(p, \alpha)$:

# adapted from https://matthew-brett.github.io/teaching/fdr.html
alpha = 0.1  # level of control on the FDR

plt.plot(sorted_pval[:1000], '.')
plt.plot(alpha * np.arange(1, 1000) / p, 'r')
plt.title("BH procedure")

below = sorted_pval < alpha * np.arange(1, p + 1) / p  
# True where p(i) < qi / N
bh_index = np.max(np.where(below)[0])
# Max Python array index  where p(i) < qi / N
print('BH statistic:', sorted_pval[bh_index])
print('BH index  :', bh_index + 1) 

# %%
# Intrinsic limitations of FDR control
# ------------------------------------


# As noted by Goeman and Solari (2011), an important limitation to the above
# approach is that FDR controlling procedures output a single set of 
# "significant features", whereas users are typically interested in other
# subsets of features.

# Another caveat is that FDR control (which is by definition a control in
# expectation), may not be interpretable in the situation where the underlying
# FDP is highly variable. Unfortunately, this situation is common in 
# high-dimensional data sets with non independent features, in particular in
# genomics and neuroimaging.

# Because we are working with simuated data, we can actually calculate the true
# False Discovery Proportion (FDP) as done in the next cell. However, this
# could not be done when analyzing actual data.

below0 = pval[:p0] < (alpha * (bh_index - 1) / p)
fp = np.sum(below0)
print('False positives:', fp)

fdp = fp / bh_index  
print('False discovery proportion:', fdp)

# %%
# 3. Post hoc inference
# ---------------------

# In order to bypass these limitations, post hoc inference makes it possible
# to *build confidence statements on the number of true/false positives within 
# any set $S$ of selected variables*: $S$ may be selected after seing the data
# (e.g., $S$ may be the set of rejections by the BH procedure), and multiple
# choices of $S$ are allowed. Post hoc inference has been popularized by 
# Goeman and Solari (2011) and its application to neuroimaging data is 
# illustrated in Rosenblatt *et al*, 2018.  
# This approach is implemented in the 'ARI'package, which relies on the
# 'cherry' R package.

# Below, we use an equivalent formulation of the bound of Goeman and Solari
# (2011) implemented in the 'sansSouci.python' Python package. 

# %%
# Upper bound on the number of false discoveries
# ----------------------------------------------

# We obtain an upper bound on $p_0$, the number of true null hypotheses, by
# taking $S=$ all $p$ hypotheses. With probability $1-\alpha = 0.9$ the number
# of true null hypotheses is less than this bound:

thr = sa.linear_template(alpha, p, p)
bound = sa.max_fp(pval, thr)
print("\nBound on the number of null variables:", int(bound))
print('Note: the number of true null hypotheses is', p0)

# %%
# Number of false positives by BH

# Post hoc inference allows the user to select any subset of features $S$ of
# interest. In particular, we can chose $S$ as the set of features selected
# by the BH procedure.

bound = sa.max_fp(sorted_pval[np.where(below)], thr)
print("GS2011 post hoc bound on false positives for BH set:", int(bound))

# In this particular example, the bound is substantially larger than the true
# number of false positives, meaning that the GS2011 bound is conservative.

# %%
# Confidence curves for the FDP

# Confidence curves for the False Discovery Proportion are another typical
# output of post hoc inference. These curves display post hoc bounds as a
# function of the number of most significant features retained. The user may
# choose how many features to retain based on the value of the bound.

max_fp = sa.curve_max_fp(sorted_pval, thr)
max_fdp = max_fp / np.arange(1, p + 1)
min_tp = np.arange(1, p + 1) - max_fp

plt.figure()
plt.plot(min_tp[:1000], )
plt.plot([bh_index, bh_index], [0, min_tp[1000]], color='k')
plt.text(bh_index + 5, min_tp[bh_index] / 2, "BH set", color='k')
plt.title("Lower confidence bound on the TP among smallest k p-values")
plt.xlabel("k")

# %%
# 4. Improved post hoc inference by adaptation to the dependence
# --------------------------------------------------------------

# As discussed in Blanchard, Neuvial, and Roquain (2020), the above-described
# bound is known to be valid only under certain positive dependence
# assumptions(PRDS) on the joint $p$-value distribution. 
# Although the PRDS assumption is widely accepted for fMRI studies 
# (see Genovese, Lazar, and Nichols (2002),Nichols and Hayasaka (2003)),
# we argue (and demonstrate below) that this
# assumption yields overly conservative post hoc bounds.
# Indeed, the Simes bound is by construction not adaptive to the specific type
# of dependence at hand for a particular data set.

# To bypass these limitations, Blanchard, Neuvial, and Roquain (2020) have
# proposed a randomization-based procedure known as $\lambda$-calibration,
# which yields tighter bounds that are adapted to the dependency observed in
# the data set at hand. We note that a related approach has been proposed by
# Hemerik, Solari, and Goeman (2019), and Andreella *et al* (2020)
# (https://arxiv.org/abs/2012.00368).  In the case of two-sample tests, this
# calibration can be achieved by permutation of class labels, which is
# available in the sansSouci.python package:

# %%
n_permutations = 1000
pval0 = sa.get_permuted_p_values(
    X,
    y,
    n_permutations=n_permutations,
    row_test_fun=sa.row_welch_tests
)
pivot_stat = sa.get_pivotal_stats(pval0, k_max=p)
lambda_ = np.quantile(pivot_stat, alpha)

# Here we obtain $\lambda > \alpha$. This practically means that in order to
# obtain post hoc statements at confidence level $1-\alpha$, the user can use
# the GS2011 bound at level $\lambda$. As  $\lambda > \alpha$, this means that
# the new proposed bound will tighter (ie, less conservative) than the original
# GS2011 bound. The gap between $\lambda$ and $\alpha$ can be interpreted as
# the power gain obtained by $\lambda$-calibration, which is illustrated below.

# %%
# Upper bound on the number of null hypotheses
# --------------------------------------------

# We obtain an upper bound on $p_0$, the number of true null hypotheses, by
# taking $S=$ all $p$ hypotheses. With probability $1-\alpha = 0.9$ the number
# of true null hypotheses should be less than this bound:

thr_cal = sa.linear_template(lambda_, p, p)
bound = sa.max_fp(pval, thr_cal)
print("\nBound on the number of null variables with BNR2020:", int(bound))

# As expected, the bound is still valid, and tighter than before.

# %%
# Number of false positives by BH
# We now calculate an upper bound on the number of false positives among the
# hypotheses rejected by the BH procedure.

bound_cal = sa.max_fp(sorted_pval[np.where(below)], thr_cal)
print("BNR2020 post hoc bound on false positives for the BH set:",
      int(bound_cal))

# The BNR2020 bound is still valid, and tighter than the GS2011 bound.

# %%
# Confidence envelopes for the FDP
# --------------------------------
max_fp_cal = sa.curve_max_fp(sorted_pval, thr_cal)
max_fdp_cal = max_fp_cal / np.arange(1, p + 1)

plt.figure()
plt.plot(max_fdp_cal[:1000], '-r', label='Simes + lambda-calibration')
plt.plot(max_fdp[:1000], '-', label='Simes')
plt.title('Upper confidence bound on the FDP among smallest k p-values')
plt.plot([bh_index, bh_index], [0, max_fdp[1000]], color='k')
plt.text(bh_index + 5, max_fdp_cal[bh_index] / 2, "BH set", color='k')
plt.xlabel('k')
plt.legend()

# The upper bound obtained by $\lambda$-calibration (in red) is uniformly 
# tighter than the original "parametric" one.

plt.show()