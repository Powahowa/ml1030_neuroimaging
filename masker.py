# %% 
from nilearn import plotting
from nilearn import image
import nilearn
import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn.image import load_img, math_img


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

# resample faces

facesSliceResamp = NiftiMasker(mask_img=facesMaskFile, target_affine=facesSlice.affine, target_shape=facesSlice.shape, standardize=True)

fitted = facesSliceResamp.fit(facesSlice)

maskedArray = facesSliceResamp.transform(facesSlice)

faces_resamp_mask = fitted.inverse_transform(X=maskedArray)

faces_resamp_mask.to_filename("sub-9001_ses-1_task-faces_resamp_mask.nii.gz")

#BINARIZING values in mask
tstat_img = load_img(faces_resamp_mask)
#supposedly: all values greater than 0 are set to 1, and all values less than 0 are set to zero.
faces_resamp_mask = math_img('img > 0', img=tstat_img)

plotMask(faces_resamp_mask, facesSlice)


#%%
# resample sleepiness

sleepinessSliceResamp = NiftiMasker(mask_img=sleepinessMaskFile, target_affine=facesSlice.affine, target_shape=facesSlice.shape, standardize=True)

fitted = sleepinessSliceResamp.fit(sleepinessSlice)
maskedArray = sleepinessSliceResamp.transform(sleepinessSlice)
sleepiness_resamp_mask = fitted.inverse_transform(X=maskedArray)

sleepiness_resamp_mask.to_filename("sub-9001_ses-1_task-sleepiness_resamp_mask.nii.gz")

#BINARIZING values in mask
tstat_img = load_img(sleepiness_resamp_mask)
#supposedly: all values greater than 0 are set to 1, and all values less than 0 are set to zero.
sleepiness_resamp_mask = math_img('img > 0', img=tstat_img)

plotMask(sleepiness_resamp_mask, facesSlice)


#%% calculate intersection of faces and sleepiness masks

intersectedFS = nilearn.masking.intersect_masks([faces_resamp_mask, sleepiness_resamp_mask], threshold=1, connected=True)


#plot intersected mask on faces slice
plotMask(intersectedFS, sleepinessSlice)

# %%

#cropping/applying mask on image attempt

cropMask = NiftiMasker(mask_img=intersectedFS, standardize=True)

fitted = cropMask.fit(sleepinessSlice)

maskedArray = cropMask.transform(sleepinessSlice)

sleepinessCrop = fitted.inverse_transform(X=maskedArray)

#cropping/applying mask on image attempt

cropMask = NiftiMasker(mask_img=intersectedFS, standardize=True)

fitted = cropMask.fit(facesSlice)

maskedArray = cropMask.transform(facesSlice)

facesCrop = fitted.inverse_transform(X=maskedArray)


# %%
nilearn.plotting.plot_img(facesCrop, cut_coords=[0,0,0], title="Masked Faces Image")
nilearn.plotting.plot_img(facesSlice, cut_coords=[0,0,0], title="Original Faces Image")
nilearn.plotting.plot_img(sleepinessCrop, cut_coords=[0,0,0], title = "Masked Sleepiness Image")
nilearn.plotting.plot_img(sleepinessSlice, cut_coords=[0,0,0], title= "Original Sleepiness Image")


# %%
