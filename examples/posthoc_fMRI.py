# Data obtained via the code of the nilearn python notebook: [Massively univariate analysis of a motor task from the Localizer dataset](https://nilearn.github.io/auto_examples/05_advanced/plot_localizer_mass_univariate_methods.html#sphx-glr-auto-examples-05-advanced-plot-localizer-mass-univariate-methods-py)

# ---------------
# setup
# ---------------
import numpy as np
import sanssouci as sa
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.image import get_data

# ---------------
# load data
# ---------------
n_samples = 16

localizer_dataset_left = datasets.fetch_localizer_contrasts(
    ["left visual click"], n_subjects=n_samples)

localizer_dataset_right = datasets.fetch_localizer_contrasts(
    ["right visual click"], n_subjects=n_samples)

# ---------------
# quality control
# ---------------

tested_var_left = localizer_dataset_left.ext_vars['pseudo']
# Quality check / Remove subjects with bad tested variate
mask_quality_check_left = np.where(tested_var_left != b'n/a')[0]
n_samples_left = mask_quality_check_left.size
contrast_map_filenames_left = [localizer_dataset_left.cmaps[i]
                          for i in mask_quality_check_left]
tested_var_left = tested_var_left[mask_quality_check_left].astype(float).reshape((-1, 1))

tested_var_right = localizer_dataset_right.ext_vars['pseudo']
# Quality check / Remove subjects with bad tested variate
mask_quality_check_right = np.where(tested_var_right != b'n/a')[0]
n_samples_right = mask_quality_check_right.size
contrast_map_filenames_right = [localizer_dataset_right.cmaps[i]
                          for i in mask_quality_check_right]
tested_var_right = tested_var_right[mask_quality_check_right].astype(float).reshape((-1, 1))

# ---------------
# smoothing
# ---------------
smt = 5

nifti_masker = NiftiMasker(
    smoothing_fwhm=smt,
    memory='nilearn_cache', memory_level=1)  # cache options

fmri_masked_left = nifti_masker.fit_transform(contrast_map_filenames_left)
fmri_masked_left.shape

fmri_masked_right = nifti_masker.fit_transform(contrast_map_filenames_right)
fmri_masked_right.shape

fmri_masked = np.concatenate((fmri_masked_left, fmri_masked_right))
fmri_masked.shape

columns_ok_left=["0"]*fmri_masked_left.shape[0]
columns_ok_right=["1"]*fmri_masked_right.shape[0]
columns_ok=columns_ok_left+columns_ok_right
categ = np.array([float(columns_ok[i]) for i in range(len(columns_ok))])
len(categ)



p = fmri_masked.shape[1]
categ


# --------------------
# Permutation p-values
# --------------------
B = 100
pval0=sa.get_perm_p(fmri_masked, categ, B=B , row_test_fun=sa.row_welch_tests)

K=p
piv_stat=sa.get_pivotal_stats(pval0, K=K)


# -----------
# Calibration
# -----------

alpha=0.1

lambda_quant=np.quantile(piv_stat, alpha)
thr=sa.t_linear(lambda_quant, np.arange(1,p+1), p)

# --------------
# Post hoc bound
# --------------

swt=sa.row_welch_tests(X, categ)
p_values=swt['p_value'][:]
pvals=p_values[:10]

bound = sa.max_fp(pvals, thr)
print(bound)

# --------------------
# Confidence envelopes
# --------------------

max_FP=sa.curve_max_fp(p_values, thr)

rg = np.arange(1, p+1)
max_FDP = max_FP / rg
min_TP = rg - max_FP
plt.subplot(121)
plt.plot(max_FDP)
plt.title('Upper bound on FDP')
plt.xlim(1, 1000)
plt.subplot(122)
plt.plot(min_TP)
plt.title('Lower bound on TP')
plt.xlim(1, 1000)
plt.show()
