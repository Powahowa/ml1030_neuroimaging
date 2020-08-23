# %% 
from nilearn import plotting
from nilearn import image
import nilearn
import nibabel as nib
import numpy as np
from nilearn.input_data import NiftiMasker
from nilearn.image import load_img, math_img
import glob #filesystem manipulation
import pandas as pd 
import pathlib
# Parellization libraries
from joblib import Parallel, delayed
import os

dataDir = "../data/preprocessed/"
subjectDir = "../data/preprocessed/sub-9001/"
sessionDir = "ses-1/"
masktestDir = "../data/masktest/"
# %% Function Definitions

#returns # of images in time dimension
def imagesinTimeDim (fullImage):
    iterable = image.iter_img(fullImage)
    return sum(1 for _ in iterable)

#plots mask file on specified image background file
def plotMask (maskFile, imageFile, affine=None):
    
    if (affine == None):
        affine = imageFile.affine

    masker = NiftiMasker(mask_img=maskFile, target_affine=affine, standardize=False)
    fmri_masked = masker.fit(imageFile)

    # Generate a report with the mask on normalized image
    report = masker.generate_report()
    return report

#load all SLICES of base scans for comparison
def loadSlice (task, indexPosition):

    if(task == "arrows"):
        slice = image.index_img(subjectDir + 
        sessionDir + 
        "func/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", indexPosition)
    if(task == "faces"):
        slice = image.index_img(subjectDir + 
        sessionDir + 
        "func/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", indexPosition)
    if(task == "hands"):
        slice = image.index_img(subjectDir + 
        sessionDir + 
        "func/sub-9001_ses-1_task-hands_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", indexPosition)
    if(task == "rest"):
        slice = image.index_img(subjectDir + 
        sessionDir + 
        "func/sub-9001_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", indexPosition)
    if(task == "sleepiness"):
        slice = image.index_img(subjectDir + 
        sessionDir + 
        "func/sub-9001_ses-1_task-sleepiness_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", indexPosition)

    return slice

def loadAllMasks ():
    #recursively add all mask files to paths list
    paths = list(pathlib.Path(dataDir).glob('**/func/*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'))
    fileNames = paths.copy()

    #remove path from fileNames leaving us just with the raw filename
    for i in range(len(fileNames)):
        fileNames[i] = os.path.basename(fileNames[i].name)

    #create dataframe from paths and filenames
    maskDF = pd.DataFrame(list(zip(paths, fileNames)), 
                            columns =['path', 'filename']) 
    print("Masks loaded:", len(maskDF['path'].tolist()))
    return maskDF

def resampleMask(maskFile, affine=None):
    #standardizing to the affine and shape of sleepiness (as the most restrictive image)
    sleepinessSliceAffine = loadSlice(task="sleepiness", indexPosition=0).affine
    sleepinessSliceShape = loadSlice(task="sleepiness", indexPosition=0).shape

    maskResamp = NiftiMasker(mask_img=maskFile, target_affine=sleepinessSliceAffine, 
    target_shape=sleepinessSliceShape, standardize=False)
    maskResamp.fit()
    #maskResamp.mask_img_.to_filename("sub-9001_ses-1_task-hands_resamp_mask.nii.gz")
    #plotMask(maskResamp.mask_img_, loadSlice(task="hands", indexPosition=0))
    return maskResamp.mask_img_

def intersectMasks():
    maskDF = loadAllMasks()
    masksList = maskDF['path'].tolist()

    #non parallel version
    #pos = 1
    # for i in masksList: 
    #     i = resampleMask(i.__str__())#str thing fixes a "windowsPath" object error.
    #     print("Mask #", pos, "resampled successfully.")
    #     pos = pos + 1
    print("--Resampling Masks--")
    masksList = Parallel(n_jobs=-1, verbose=100)(delayed(resampleMask)(i.__str__()) for i in masksList)

    #calculate interset of the masks. threshold = 1 means intersection, not union
    intersectedMask = nilearn.masking.intersect_masks(masksList, threshold=1, connected=True)
    return intersectedMask, masksList

#%% get final intersected mask from all available masks

finalMask, maskList = intersectMasks()
print ("\nMaskList:", maskList)
print("\n# of masks intersected:", len(maskList))
print ("\nFinal Intersected Mask:")
#plot intersected mask on sleepiness slice
plotMask(finalMask, loadSlice(task="sleepiness", indexPosition=0))
finalMask.to_filename("sub-9001-9072_resamp_intersected_mask.nii.gz")

# %%

#cropping/applying mask on image attempt for each task (final mask)

savePlots = False

#arrows
cropMask = NiftiMasker(mask_img="sub-9001-9072_resamp_intersected_mask.nii.gz", standardize=False)
#fitted = cropMask.fit(loadSlice(task="arrows", indexPosition=0))
#maskedArray = cropMask.transform(loadSlice(task="arrows", indexPosition=0))
#above 2 lines replaced by "fit_transform"
maskedArray = cropMask.fit_transform(loadSlice(task="arrows", indexPosition=0))
arrowsCrop = cropMask.inverse_transform(X=maskedArray)
#this is just a slice, no point saving it to disk
#arrowsCrop.to_filename("sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-preproc_masked_(sub-9001-9072_resamp_intersected)_bold.nii.gz")

plt = nilearn.plotting.plot_img(loadSlice("arrows", 0), cut_coords=[0,0,0], title="Original arrows Image")
if savePlots == True:
    plt.savefig('../plots/masking/sub-9001-9072_resamp_intersected/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-preproc_original_bold.png')
    plt.close()
plt = nilearn.plotting.plot_img(arrowsCrop, cut_coords=[0,0,0], title="Masked arrows Image")
if savePlots == True:
    plt.savefig('../plots/masking/sub-9001-9072_resamp_intersected/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-preproc_masked_(sub-9001-9072_resamp_intersected)_bold.png')
    plt.close()

#faces
cropMask = NiftiMasker(mask_img="sub-9001-9072_resamp_intersected_mask.nii.gz", standardize=False)
#fitted = cropMask.fit(loadSlice(task="faces", indexPosition=0))
#maskedArray = cropMask.transform(loadSlice(task="faces", indexPosition=0))
#above 2 lines replaced by "fit_transform"
maskedArray = cropMask.fit_transform(loadSlice(task="faces", indexPosition=0))
facesCrop = cropMask.inverse_transform(X=maskedArray)
#this is just a slice, no point saving it to disk
#facesCrop.to_filename("sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_masked_(sub-9001-9072_resamp_intersected)_bold.nii.gz")

plt = nilearn.plotting.plot_img(loadSlice("faces", 0), cut_coords=[0,0,0], title="Original faces Image")
if savePlots == True:
    plt.savefig('../plots/masking/sub-9001-9072_resamp_intersected/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_original_bold.png')
    plt.close()
plt = nilearn.plotting.plot_img(facesCrop, cut_coords=[0,0,0], title="Masked faces Image")
if savePlots == True:
    plt.savefig('../plots/masking/sub-9001-9072_resamp_intersected/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_masked_(sub-9001-9072_resamp_intersected)_bold.png')
    plt.close()

#hands
cropMask = NiftiMasker(mask_img="sub-9001-9072_resamp_intersected_mask.nii.gz", standardize=False)
#fitted = cropMask.fit(loadSlice(task="hands", indexPosition=0))
#maskedArray = cropMask.transform(loadSlice(task="hands", indexPosition=0))
#above 2 lines replaced by "fit_transform"
maskedArray = cropMask.fit_transform(loadSlice(task="hands", indexPosition=0))
handsCrop = cropMask.inverse_transform(X=maskedArray)
#this is just a slice, no point saving it to disk
#handsCrop.to_filename("sub-9001_ses-1_task-hands_space-MNI152NLin2009cAsym_desc-preproc_masked_(sub-9001-9072_resamp_intersected)_bold.nii.gz")

plt = nilearn.plotting.plot_img(loadSlice("hands", 0), cut_coords=[0,0,0], title="Original hands Image")
if savePlots == True:
    plt.savefig('../plots/masking/sub-9001-9072_resamp_intersected/sub-9001_ses-1_task-hands_space-MNI152NLin2009cAsym_desc-preproc_original_bold.png')
    plt.close()
plt = nilearn.plotting.plot_img(handsCrop, cut_coords=[0,0,0], title="Masked hands Image")
if savePlots == True:
    plt.savefig('../plots/masking/sub-9001-9072_resamp_intersected/sub-9001_ses-1_task-hands_space-MNI152NLin2009cAsym_desc-preproc_masked_(sub-9001-9072_resamp_intersected)_bold.png')
    plt.close()

#rest
cropMask = NiftiMasker(mask_img="sub-9001-9072_resamp_intersected_mask.nii.gz", standardize=False)
#fitted = cropMask.fit(loadSlice(task="rest", indexPosition=0))
#maskedArray = cropMask.transform(loadSlice(task="rest", indexPosition=0))
#above 2 lines replaced by "fit_transform"
maskedArray = cropMask.fit_transform(loadSlice(task="rest", indexPosition=0))
restCrop = cropMask.inverse_transform(X=maskedArray)
#this is just a slice, no point saving it to disk
#restCrop.to_filename("sub-9001_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-preproc_masked_(sub-9001-9072_resamp_intersected)_bold.nii.gz")

plt = nilearn.plotting.plot_img(loadSlice("rest", 0), cut_coords=[0,0,0], title="Original rest Image")
if savePlots == True:
    plt.savefig('../plots/masking/sub-9001-9072_resamp_intersected/sub-9001_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-preproc_original_bold.png')
    plt.close()
plt = nilearn.plotting.plot_img(restCrop, cut_coords=[0,0,0], title="Masked rest Image")
if savePlots == True:
    plt.savefig('../plots/masking/sub-9001-9072_resamp_intersected/sub-9001_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-preproc_masked_(sub-9001-9072_resamp_intersected)_bold.png')
    plt.close()

#sleepiness
cropMask = NiftiMasker(mask_img="sub-9001-9072_resamp_intersected_mask.nii.gz", standardize=False)
#fitted = cropMask.fit(loadSlice(task="sleepiness", indexPosition=0))
#maskedArray = cropMask.transform(loadSlice(task="sleepiness", indexPosition=0))
#above 2 lines replaced by "fit_transform"
maskedArray = cropMask.fit_transform(loadSlice(task="sleepiness", indexPosition=0))
sleepinessCrop = cropMask.inverse_transform(X=maskedArray)
#this is just a slice, no point saving it to disk
#sleepinessCrop.to_filename("sub-9001_ses-1_task-sleepiness_space-MNI152NLin2009cAsym_desc-preproc_masked_(sub-9001-9072_resamp_intersected)_bold.nii.gz")

plt = nilearn.plotting.plot_img(loadSlice("sleepiness", 0), cut_coords=[0,0,0], title="Original sleepiness Image")
if savePlots == True:
    plt.savefig('../plots/masking/sub-9001-9072_resamp_intersected/sub-9001_ses-1_task-sleepiness_space-MNI152NLin2009cAsym_desc-preproc_original_bold.png')
    plt.close()
plt = nilearn.plotting.plot_img(sleepinessCrop, cut_coords=[0,0,0], title="Masked sleepiness Image")
if savePlots == True:
    plt.savefig('../plots/masking/sub-9001-9072_resamp_intersected/sub-9001_ses-1_task-sleepiness_space-MNI152NLin2009cAsym_desc-preproc_masked_(sub-9001-9072_resamp_intersected)_bold.png')
    plt.close()

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


#%% load all MASKS of base scans

arrowsMaskFile = (subjectDir + 
    sessionDir + 
    "func/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")

facesMaskFile = (subjectDir + 
    sessionDir + 
    "func/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
    
handsMaskFile = (subjectDir + 
    sessionDir + 
    "func/sub-9001_ses-1_task-hands_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
    
restMaskFile = (subjectDir + 
    sessionDir + 
    "func/sub-9001_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
    
sleepinessMaskFile = (subjectDir + 
    sessionDir + 
    "func/sub-9001_ses-1_task-sleepiness_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")

#%% plot mask of faces on faces slice

plotMask(facesMaskFile, loadSlice(task="faces", indexPosition=0))

#%% plot mask of arrows on faces slice

plotMask(arrowsMaskFile, loadSlice(task="faces", indexPosition=0))

#%% plot mask of sleepiness on sleepiness slice

plotMask(sleepinessMaskFile, loadSlice(task="sleepiness", indexPosition=0))

#%% calculate intersection of faces and arrows masks

intersectedFA = nilearn.masking.intersect_masks([facesMaskFile, arrowsMaskFile], threshold=1, connected=True)

#plot intersected mask on faces slice
plotMask(intersectedFA, loadSlice(task="faces", indexPosition=0))

# %%

#standardizing to the affine and shape of sleepiness (as the most restrictive image)

sleepinessSliceAffine = loadSlice(task="sleepiness", indexPosition=0).affine
sleepinessSliceShape = loadSlice(task="sleepiness", indexPosition=0).shape

#then resample all masks using NiftiMasker

#%%
#arrows resample
arrowsMaskResamp = NiftiMasker(mask_img=arrowsMaskFile, target_affine=sleepinessSliceAffine, 
target_shape=sleepinessSliceShape, standardize=False)
arrowsMaskResamp.fit()
arrowsMaskResamp.mask_img_.to_filename("sub-9001_ses-1_task-arrows_resamp_mask.nii.gz")
plotMask(arrowsMaskResamp.mask_img_, loadSlice(task="arrows", indexPosition=0))

#%%
#faces resample
facesMaskResamp = NiftiMasker(mask_img=facesMaskFile, target_affine=sleepinessSliceAffine, 
target_shape=sleepinessSliceShape, standardize=False)
facesMaskResamp.fit()
facesMaskResamp.mask_img_.to_filename("sub-9001_ses-1_task-faces_resamp_mask.nii.gz")
plotMask(facesMaskResamp.mask_img_, loadSlice(task="faces", indexPosition=0))

#%%
#hands resample
handsMaskResamp = NiftiMasker(mask_img=handsMaskFile, target_affine=sleepinessSliceAffine, 
target_shape=sleepinessSliceShape, standardize=False)
handsMaskResamp.fit()
handsMaskResamp.mask_img_.to_filename("sub-9001_ses-1_task-hands_resamp_mask.nii.gz")
plotMask(handsMaskResamp.mask_img_, loadSlice(task="hands", indexPosition=0))

#%%
#rest resample
restMaskResamp = NiftiMasker(mask_img=restMaskFile, target_affine=sleepinessSliceAffine, 
target_shape=sleepinessSliceShape, standardize=False)
restMaskResamp.fit()
restMaskResamp.mask_img_.to_filename("sub-9001_ses-1_task-rest_resamp_mask.nii.gz")
plotMask(restMaskResamp.mask_img_, loadSlice(task="rest", indexPosition=0))

#%%
#sleepiness resample
sleepinessMaskResamp = NiftiMasker(mask_img=sleepinessMaskFile, target_affine=sleepinessSliceAffine, 
target_shape=sleepinessSliceShape, standardize=False)
sleepinessMaskResamp.fit()
sleepinessMaskResamp.mask_img_.to_filename("sub-9001_ses-1_task-sleepiness_resamp_mask.nii.gz")
plotMask(sleepinessMaskResamp.mask_img_, loadSlice(task="sleepiness", indexPosition=0))


#%% calculate intersection of all 5 masks

resampledMasks = [arrowsMaskResamp.mask_img_, facesMaskResamp.mask_img_,
 handsMaskResamp.mask_img_, restMaskResamp.mask_img_, sleepinessMaskResamp.mask_img_]

#calculate interset of the masks. threshold = 1 means intersection, not union
intersectedMask = nilearn.masking.intersect_masks(resampledMasks, threshold=1, connected=True)

#plot intersected mask on sleepiness slice
plotMask(intersectedMask, loadSlice(task="sleepiness", indexPosition=0))


# %%

#cropping/applying mask on image attempt for each task

#arrows
cropMask = NiftiMasker(mask_img=intersectedMask, standardize=False)
#fitted = cropMask.fit(loadSlice(task="arrows", indexPosition=0))
#maskedArray = cropMask.transform(loadSlice(task="arrows", indexPosition=0))
#above 2 lines replaced by "fit_transform"
maskedArray = cropMask.fit_transform(loadSlice(task="arrows", indexPosition=0))
arrowsCrop = cropMask.inverse_transform(X=maskedArray)

plt = nilearn.plotting.plot_img(loadSlice("arrows", 0), cut_coords=[0,0,0], title="Original arrows Image")
#plt.savefig('../plots/masking/Original arrows Image.png')
#plt.close()
plt = nilearn.plotting.plot_img(arrowsCrop, cut_coords=[0,0,0], title="Masked arrows Image")
#plt.savefig('../plots/masking/Masked arrows Image.png')
#plt.close()

#faces
cropMask = NiftiMasker(mask_img=intersectedMask, standardize=False)
#fitted = cropMask.fit(loadSlice(task="faces", indexPosition=0))
#maskedArray = cropMask.transform(loadSlice(task="faces", indexPosition=0))
#above 2 lines replaced by "fit_transform"
maskedArray = cropMask.fit_transform(loadSlice(task="faces", indexPosition=0))
facesCrop = cropMask.inverse_transform(X=maskedArray)

plt = nilearn.plotting.plot_img(loadSlice("faces", 0), cut_coords=[0,0,0], title="Original faces Image")
#plt.savefig('../plots/masking/Original faces Image.png')
#plt.close()
plt = nilearn.plotting.plot_img(facesCrop, cut_coords=[0,0,0], title="Masked faces Image")
#plt.savefig('../plots/masking/Masked faces Image.png')
#plt.close()

#hands
cropMask = NiftiMasker(mask_img=intersectedMask, standardize=False)
#fitted = cropMask.fit(loadSlice(task="hands", indexPosition=0))
#maskedArray = cropMask.transform(loadSlice(task="hands", indexPosition=0))
#above 2 lines replaced by "fit_transform"
maskedArray = cropMask.fit_transform(loadSlice(task="hands", indexPosition=0))
handsCrop = cropMask.inverse_transform(X=maskedArray)

plt = nilearn.plotting.plot_img(loadSlice("hands", 0), cut_coords=[0,0,0], title="Original hands Image")
#plt.savefig('../plots/masking/Original hands Image.png')
#plt.close()
plt = nilearn.plotting.plot_img(handsCrop, cut_coords=[0,0,0], title="Masked hands Image")
#plt.savefig('../plots/masking/Masked hands Image.png')
#plt.close()

#rest
cropMask = NiftiMasker(mask_img=intersectedMask, standardize=False)
#fitted = cropMask.fit(loadSlice(task="rest", indexPosition=0))
#maskedArray = cropMask.transform(loadSlice(task="rest", indexPosition=0))
#above 2 lines replaced by "fit_transform"
maskedArray = cropMask.fit_transform(loadSlice(task="rest", indexPosition=0))
restCrop = cropMask.inverse_transform(X=maskedArray)

plt = nilearn.plotting.plot_img(loadSlice("rest", 0), cut_coords=[0,0,0], title="Original rest Image")
#plt.savefig('../plots/masking/Original rest Image.png')
#plt.close()
plt = nilearn.plotting.plot_img(restCrop, cut_coords=[0,0,0], title="Masked rest Image")
#plt.savefig('../plots/masking/Masked rest Image.png')
#plt.close()

#sleepiness
cropMask = NiftiMasker(mask_img=intersectedMask, standardize=False)
#fitted = cropMask.fit(loadSlice(task="sleepiness", indexPosition=0))
#maskedArray = cropMask.transform(loadSlice(task="sleepiness", indexPosition=0))
#above 2 lines replaced by "fit_transform"
maskedArray = cropMask.fit_transform(loadSlice(task="sleepiness", indexPosition=0))
sleepinessCrop = cropMask.inverse_transform(X=maskedArray)

plt = nilearn.plotting.plot_img(loadSlice("sleepiness", 0), cut_coords=[0,0,0], title="Original sleepiness Image")
#plt.savefig('../plots/masking/Original sleepiness Image.png')
#plt.close()
plt = nilearn.plotting.plot_img(sleepinessCrop, cut_coords=[0,0,0], title="Masked sleepiness Image")
#plt.savefig('../plots/masking/Masked sleepiness Image.png')
#plt.close()






#%% Old method of resampling masks, fitting, transforming, inverse transforming, binarizing then saving.

#replaced by accessing the re-sampled mask directly using [NiftiMasker object].mask_img_

# sleepinessMaskResamp = NiftiMasker(mask_img=sleepinessMaskFile, target_affine=facesSlice.affine, target_shape=facesSlice.shape, standardize=False)

# fitted = sleepinessMaskResamp.fit(sleepinessSlice)
# maskedArray = sleepinessMaskResamp.transform(sleepinessSlice)
# sleepiness_resamp_mask = fitted.inverse_transform(X=maskedArray)

# sleepiness_resamp_mask.to_filename("sub-9001_ses-1_task-sleepiness_resamp_mask.nii.gz")

# #BINARIZING values in mask
# tstat_img = load_img(sleepiness_resamp_mask)
# #supposedly: all values greater than 0 are set to 1, and all values less than 0 are set to zero.
# sleepiness_resamp_mask = math_img('img > 0', img=tstat_img)

# plotMask(sleepiness_resamp_mask, facesSlice)





# %% get unique values in raw nifti data array as a sanity check

# img = image.load_img(subjectDir + 
# sessionDir + 
# "func/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
# img.uncache()

# image_data = np.asarray(img.dataobj)

# print("unique values of actual Nitfi image")
# print(np.unique(image_data))


# img = image.load_img(subjectDir + 
#     sessionDir + 
#     "func/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")

# image_data = np.asarray(img.dataobj)

# img.uncache()

# print("unique values of actual MASK image")
# print(np.unique(image_data))



#%% look at all the image shapes

# shapes = []

# img = image.load_img(subjectDir + 
# sessionDir + 
# "func/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

# shapes.append(img.shape)
# img.uncache()

# img = image.load_img(subjectDir + 
# sessionDir + 
# "func/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

# shapes.append(img.shape)
# img.uncache()

# img = image.load_img(subjectDir + 
# sessionDir + 
# "func/sub-9001_ses-1_task-hands_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

# shapes.append(img.shape)
# img.uncache()

# img = image.load_img(subjectDir + 
# sessionDir + 
# "func/sub-9001_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

# shapes.append(img.shape)
# img.uncache()

# img = image.load_img(subjectDir + 
# sessionDir + 
# "func/sub-9001_ses-1_task-sleepiness_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

# shapes.append(img.shape)
# img.uncache()

# print("Arrows, Faces, Hands, Rest, Sleepiness")
# shapes