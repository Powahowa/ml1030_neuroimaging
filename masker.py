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

# %% Function Definitions

#returns # of images in time dimension
def imagesinTimeDim (fullImage):
    iterable = image.iter_img(fullImage)
    return sum(1 for _ in iterable)

#plots mask file on specified image background file
def plotMask (maskFile, imageFile, affine=None):
    
    if (affine == None):
        affine = imageFile.affine

    masker = NiftiMasker(mask_img=maskFile, target_affine=affine, standardize=True)
    fmri_masked = masker.fit(imageFile)

    # Generate a report with the mask on normalized image
    report = masker.generate_report()
    return report

def makeNiftiMask (maskFile, imageFile, affine=None):
    
    if (affine == None):
        affine = imageFile.affine

    masker = NiftiMasker(mask_img=maskFile, target_affine=affine, standardize=True)

    return masker
#%%

#load all FULL base scans for comparison (huge amount of memory required, 1 or 2 at a time)

# arrowsFull = image.load_img(subjectDir + 
# sessionDir + 
# "func/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

# facesFull = image.load_img(subjectDir + 
# sessionDir + 
# "func/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

# handsFull = image.load_img(subjectDir + 
# sessionDir + 
# "func/sub-9001_ses-1_task-hands_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

# restFull = image.load_img(subjectDir + 
# sessionDir + 
# "func/sub-9001_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

# sleepinessFull = image.load_img(subjectDir + 
# sessionDir + 
# "func/sub-9001_ses-1_task-sleepiness_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

#%%

#load all SLICES of base scans for comparison

#specifies where to take the slice
indexPosition = 0

arrowsSlice = image.index_img(subjectDir + 
sessionDir + 
"func/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", indexPosition)

facesSlice = image.index_img(subjectDir + 
sessionDir + 
"func/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", indexPosition)

# handsSlice = image.index_img(subjectDir + 
# sessionDir + 
# "func/sub-9001_ses-1_task-hands_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", indexPosition)

# restSlice = image.index_img(subjectDir + 
# sessionDir + 
# "func/sub-9001_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", indexPosition)

sleepinessSlice = image.index_img(subjectDir + 
sessionDir + 
"func/sub-9001_ses-1_task-sleepiness_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", indexPosition)



#%% load all MASKS of base scans

arrowsMaskFile = (subjectDir + 
    sessionDir + 
    "func/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")

facesMaskFile = (subjectDir + 
    sessionDir + 
    "func/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
    
# handsMaskFile = (subjectDir + 
#     sessionDir + 
#     "func/sub-9001_ses-1_task-hands_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
    
# restMaskFile = (subjectDir + 
#     sessionDir + 
#     "func/sub-9001_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
    
sleepinessMaskFile = (subjectDir + 
    sessionDir + 
    "func/sub-9001_ses-1_task-sleepiness_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")


#%% plot mask of faces on faces slice

plotMask(facesMaskFile, facesSlice)

#%% plot mask of arrows on faces slice

plotMask(arrowsMaskFile, facesSlice)

#%% plot mask of sleepiness on sleepiness slice

plotMask(sleepinessMaskFile, facesSlice)

#%% calculate intersection of faces and arrows masks

intersectedFA = nilearn.masking.intersect_masks([facesMaskFile, arrowsMaskFile], threshold=1, connected=True)

#plot intersected mask on faces slice
plotMask(intersectedFA, facesSlice)

# %%

#resample masks using NiftiMasker

facesSliceResamp = makeNiftiMask(facesMaskFile, facesSlice)

sleepinessSliceResamp = makeNiftiMask(sleepinessMaskFile, facesSlice)

fitted = facesSliceResamp.fit(facesSlice)

facesSliceResampNifti = NiftiMasker.inverse_transform(facesSliceResamp, X=fitted)


#%% calculate intersection of faces and sleepiness masks

intersectedFS = nilearn.masking.intersect_masks([facesSliceResamp, sleepinessSliceResamp], threshold=1, connected=True)


#plot intersected mask on faces slice
plotMask(intersectedFA, sleepinessSlice)

# %%
