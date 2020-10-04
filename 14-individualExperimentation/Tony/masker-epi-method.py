# %%
from nilearn import plotting
from nilearn import image
import nilearn

# %%
subjectDir = "../data/preprocessed/sub-9001/"
sessionDir = "ses-1/"

# %%
faces_img = image.index_img(subjectDir + 
    sessionDir + 
    "func/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 0)
rest_img = image.index_img(subjectDir + 
    sessionDir + 
    "func/sub-9001_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 0)
rest_img = image.resample_img(rest_img, faces_img.affine, faces_img.shape)

# %%
faces_mask = nilearn.masking.compute_epi_mask(faces_img)
masker = NiftiMasker(mask_img=faces_mask)
masker.fit(rest_img)
report = masker.generate_report()
report

# %%
rest_mask = nilearn.masking.compute_epi_mask(rest_img)
masker = NiftiMasker(mask_img=rest_mask)
masker.fit(rest_img)
report = masker.generate_report()
report

# %%
intersected_mask = nilearn.masking.compute_multi_epi_mask(
    [faces_img, rest_img], target_affine=faces_img.affine, threshold=1)
masker = NiftiMasker(mask_img=intersected_mask)
masker.fit(rest_img)
report = masker.generate_report()
report

# %%