# %% 
from nilearn import plotting
from nilearn import image
import nilearn
import nibabel as nib
from nilearn.input_data import NiftiMasker


# %%
subjectDir = "./data/preprocessed/sub-9001/"

sessionDir = "ses-1/"

# %%
# Plot anatomy image
plotting.plot_anat(subjectDir + 
    "anat/sub-9001_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz")


# %%
# How many images in the time dimension?

iterable = image.iter_img(subjectDir + 
sessionDir + 
"func/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
sum(1 for _ in iterable)

#%%

#load two FULL base scans for comparison

# sleepinessFull = image.load_img(subjectDir + 
# sessionDir + 
# "func/sub-9001_ses-1_task-sleepiness_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

# facesFull = image.load_img(subjectDir + 
# sessionDir + 
# "func/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")


#%%

#load two SLICES of base scans for comparison

facesSlice = image.index_img(subjectDir + 
sessionDir + 
"func/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 0)

sleepinessSlice = image.index_img(subjectDir + 
sessionDir + 
"func/sub-9001_ses-1_task-sleepiness_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 0)

arrowsSlice = image.index_img(subjectDir + 
sessionDir + 
"func/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 0)


#%% generate mask of faces and plot on sleepiness slice

from nilearn.input_data import NiftiMasker

# plotting.plot_anat(subjectDir + 
#     "anat/sub-9001_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz")


facesMaskFile = (subjectDir + 
sessionDir + 
"func/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
maskerFaces = NiftiMasker(mask_img=facesMaskFile, standardize=True)
fmri_masked = maskerFaces.fit(facesSlice)

# Generate a report with the mask on normalized image
report = maskerFaces.generate_report()
report

#%% generate mask of sleepiness and plot on sleepiness slice

sleepinessMaskFile = (subjectDir + 
sessionDir + 
"func/sub-9001_ses-1_task-sleepiness_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
maskerSleepiness = NiftiMasker(mask_img=sleepinessMaskFile, standardize=True)
fmri_masked = maskerSleepiness.fit(facesSlice)

# Generate a report with the mask on normalized image
report = maskerSleepiness.generate_report()
report


#%% generate mask of arrows and plot on sleepiness slice

arrowsMaskFile = (subjectDir + 
sessionDir + 
"func/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
maskerarrows = NiftiMasker(mask_img=arrowsMaskFile, standardize=True)
fmri_masked = maskerarrows.fit(facesSlice)

# Generate a report with the mask on normalized image
report = maskerarrows.generate_report()
report

#%% calculate intersection of faces and sleepiness masks

#INTERSECT

intersected = nilearn.masking.intersect_masks([facesMaskFile, arrowsMaskFile], threshold=1, connected=True)



#%% plot intersected mask on faces slice

# arrowsMaskFile = (subjectDir + 
# sessionDir + 
# "func/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
maskerarrows = NiftiMasker(mask_img=intersected, standardize=True)
fmri_masked = maskerarrows.fit(arrowsSlice)

# Generate a report with the mask on normalized image
report = maskerarrows.generate_report()
report


# %%

#resampling bullshit

facesSliceResamp = nilearn.image.resample_img(facesSlice, target_affine=facesSlice.affine)

sleepinessSliceResamp = nilearn.image.resample_img(facesSlice, target_affine=facesSlice.affine)


# %%
