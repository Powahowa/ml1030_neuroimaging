# %% 
from nilearn import plotting
from nilearn import image
import nilearn
from nilearn.input_data import NiftiMasker
from nilearn.image import load_img, math_img
import glob #filesystem manipulation
import pandas as pd 
import pathlib
# Parellization libraries
from joblib import Parallel, delayed
import os
import matplotlib as plt

plt.use('Agg')

#CAMH SCC
dataDir = "/external/rprshnas01/netdata_kcni/edlab/ds000201_preproc/data/derived/fmriprep/"
subjectDir = "/external/rprshnas01/netdata_kcni/edlab/ds000201_preproc/data/derived/fmriprep/sub-9001/"
sessionDir = "ses-1/"
saveDir = "/external/rprshnas01/netdata_kcni/edlab/temp_dataknights/finalMask/"

# #CAMH SCC TEST
#dataDir = "/external/rprshnas01/netdata_kcni/edlab/temp_dataknights/dataSample/ds000201_preproc/data/derived/fmriprep/"
#subjectDir = "/external/rprshnas01/netdata_kcni/edlab/temp_dataknights/dataSample/ds000201_preproc/data/derived/fmriprep/sub-9001/"
#sessionDir = "ses-1/"
#saveDir = "/external/rprshnas01/netdata_kcni/edlab/temp_dataknights/finalMask/"

# #local - Patrick
# dataDir = "/git/ml1030_neuroimaging/data/preprocessed/"
# subjectDir = "/git/ml1030_neuroimaging/data/preprocessed/sub-9001/"
# sessionDir = "ses-1/"
# saveDir = "/git/ml1030_neuroimaging/masking/finalMask/"

os.chdir(saveDir)
print("Working Directory: ")
print(os.getcwd())
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

    # #split masks into two parts and run intersection on that to save memory?
    # length = len(masksList)
    # middle_index = length//2
    # masksList1 = masksList[:middle_index]
    # masksList2 = masksList[middle_index:]
    # intersectedMask1 = nilearn.masking.intersect_masks(masksList1, threshold=1, connected=True)
    # intersectedMask2 = nilearn.masking.intersect_masks(masksList2, threshold=1, connected=True)
    # intersectedMaskFinal = nilearn.masking.intersect_masks([intersectedMask1, intersectedMask2], threshold=1, connected=True)

    return intersectedMask, masksList


#%% get final intersected mask from all available masks

finalMask, maskList = intersectMasks()
print ("\nMaskList:", maskList)
print("\n# of masks intersected:", len(maskList))
print ("\nFinal Intersected Mask:")
finalMask.to_filename(saveDir + "final_resamp_intersected_mask.nii.gz")

# %%

#cropping/applying mask on image attempt for each task (final mask)

savePlots = True

#arrows
cropMask = NiftiMasker(mask_img="final_resamp_intersected_mask.nii.gz", standardize=False)
#fitted = cropMask.fit(loadSlice(task="arrows", indexPosition=0))
#maskedArray = cropMask.transform(loadSlice(task="arrows", indexPosition=0))
#above 2 lines replaced by "fit_transform"
maskedArray = cropMask.fit_transform(loadSlice(task="arrows", indexPosition=0))
arrowsCrop = cropMask.inverse_transform(X=maskedArray)
#this is just a slice, no point saving it to disk
#arrowsCrop.to_filename("sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-preproc_masked_(final_resamp_intersected)_bold.nii.gz")

plt = nilearn.plotting.plot_img(loadSlice("arrows", 0), cut_coords=[0,0,0], title="Original arrows Image")
if savePlots == True:
    plt.savefig(saveDir + 'plots/masking/final_resamp_intersected/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-preproc_original_bold.png')
    plt.close()
plt = nilearn.plotting.plot_img(arrowsCrop, cut_coords=[0,0,0], title="Masked arrows Image")
if savePlots == True:
    plt.savefig(saveDir + 'plots/masking/final_resamp_intersected/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-preproc_masked_(final_resamp_intersected)_bold.png')
    plt.close()

#faces
cropMask = NiftiMasker(mask_img="final_resamp_intersected_mask.nii.gz", standardize=False)
#fitted = cropMask.fit(loadSlice(task="faces", indexPosition=0))
#maskedArray = cropMask.transform(loadSlice(task="faces", indexPosition=0))
#above 2 lines replaced by "fit_transform"
maskedArray = cropMask.fit_transform(loadSlice(task="faces", indexPosition=0))
facesCrop = cropMask.inverse_transform(X=maskedArray)
#this is just a slice, no point saving it to disk
#facesCrop.to_filename("sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_masked_(final_resamp_intersected)_bold.nii.gz")

plt = nilearn.plotting.plot_img(loadSlice("faces", 0), cut_coords=[0,0,0], title="Original faces Image")
if savePlots == True:
    plt.savefig(saveDir + 'plots/masking/final_resamp_intersected/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_original_bold.png')
    plt.close()
plt = nilearn.plotting.plot_img(facesCrop, cut_coords=[0,0,0], title="Masked faces Image")
if savePlots == True:
    plt.savefig(saveDir + 'plots/masking/final_resamp_intersected/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_masked_(final_resamp_intersected)_bold.png')
    plt.close()

#hands
cropMask = NiftiMasker(mask_img="final_resamp_intersected_mask.nii.gz", standardize=False)
#fitted = cropMask.fit(loadSlice(task="hands", indexPosition=0))
#maskedArray = cropMask.transform(loadSlice(task="hands", indexPosition=0))
#above 2 lines replaced by "fit_transform"
maskedArray = cropMask.fit_transform(loadSlice(task="hands", indexPosition=0))
handsCrop = cropMask.inverse_transform(X=maskedArray)
#this is just a slice, no point saving it to disk
#handsCrop.to_filename("sub-9001_ses-1_task-hands_space-MNI152NLin2009cAsym_desc-preproc_masked_(final_resamp_intersected)_bold.nii.gz")

plt = nilearn.plotting.plot_img(loadSlice("hands", 0), cut_coords=[0,0,0], title="Original hands Image")
if savePlots == True:
    plt.savefig(saveDir + 'plots/masking/final_resamp_intersected/sub-9001_ses-1_task-hands_space-MNI152NLin2009cAsym_desc-preproc_original_bold.png')
    plt.close()
plt = nilearn.plotting.plot_img(handsCrop, cut_coords=[0,0,0], title="Masked hands Image")
if savePlots == True:
    plt.savefig(saveDir + 'plots/masking/final_resamp_intersected/sub-9001_ses-1_task-hands_space-MNI152NLin2009cAsym_desc-preproc_masked_(final_resamp_intersected)_bold.png')
    plt.close()

#rest
cropMask = NiftiMasker(mask_img="final_resamp_intersected_mask.nii.gz", standardize=False)
#fitted = cropMask.fit(loadSlice(task="rest", indexPosition=0))
#maskedArray = cropMask.transform(loadSlice(task="rest", indexPosition=0))
#above 2 lines replaced by "fit_transform"
maskedArray = cropMask.fit_transform(loadSlice(task="rest", indexPosition=0))
restCrop = cropMask.inverse_transform(X=maskedArray)
#this is just a slice, no point saving it to disk
#restCrop.to_filename("sub-9001_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-preproc_masked_(final_resamp_intersected)_bold.nii.gz")

plt = nilearn.plotting.plot_img(loadSlice("rest", 0), cut_coords=[0,0,0], title="Original rest Image")
if savePlots == True:
    plt.savefig(saveDir + 'plots/masking/final_resamp_intersected/sub-9001_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-preproc_original_bold.png')
    plt.close()
plt = nilearn.plotting.plot_img(restCrop, cut_coords=[0,0,0], title="Masked rest Image")
if savePlots == True:
    plt.savefig(saveDir + 'plots/masking/final_resamp_intersected/sub-9001_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-preproc_masked_(final_resamp_intersected)_bold.png')
    plt.close()

#sleepiness
cropMask = NiftiMasker(mask_img="final_resamp_intersected_mask.nii.gz", standardize=False)
#fitted = cropMask.fit(loadSlice(task="sleepiness", indexPosition=0))
#maskedArray = cropMask.transform(loadSlice(task="sleepiness", indexPosition=0))
#above 2 lines replaced by "fit_transform"
maskedArray = cropMask.fit_transform(loadSlice(task="sleepiness", indexPosition=0))
sleepinessCrop = cropMask.inverse_transform(X=maskedArray)
#this is just a slice, no point saving it to disk
#sleepinessCrop.to_filename("sub-9001_ses-1_task-sleepiness_space-MNI152NLin2009cAsym_desc-preproc_masked_(final_resamp_intersected)_bold.nii.gz")

plt = nilearn.plotting.plot_img(loadSlice("sleepiness", 0), cut_coords=[0,0,0], title="Original sleepiness Image")
if savePlots == True:
    plt.savefig(saveDir + 'plots/masking/final_resamp_intersected/sub-9001_ses-1_task-sleepiness_space-MNI152NLin2009cAsym_desc-preproc_original_bold.png')
    plt.close()
plt = nilearn.plotting.plot_img(sleepinessCrop, cut_coords=[0,0,0], title="Masked sleepiness Image")
if savePlots == True:
    plt.savefig(saveDir + 'plots/masking/final_resamp_intersected/sub-9001_ses-1_task-sleepiness_space-MNI152NLin2009cAsym_desc-preproc_masked_(final_resamp_intersected)_bold.png')
    plt.close()
# %%
