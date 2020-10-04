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
import nibabel as nib

plt.use('Agg')

#CAMH SCC
dataDir = "/external/rprshnas01/netdata_kcni/edlab/ds000201_preproc/data/derived/fmriprep/"
subjectDir = "/external/rprshnas01/netdata_kcni/edlab/ds000201_preproc/data/derived/fmriprep/sub-9001/"
sessionDir = "ses-1/"
saveDir = "/external/rprshnas01/netdata_kcni/edlab/temp_dataknights/"

# #CAMH SCC TEST
#dataDir = "/external/rprshnas01/netdata_kcni/edlab/temp_dataknights/dataSample/ds000201_preproc/data/derived/fmriprep/"
#subjectDir = "/external/rprshnas01/netdata_kcni/edlab/temp_dataknights/dataSample/ds000201_preproc/data/derived/fmriprep/sub-9001/"
#sessionDir = "ses-1/"
#saveDir = "/external/rprshnas01/netdata_kcni/edlab/temp_dataknights/finalMask/"

# #local - Patrick
# dataDir = "/git/ml1030_neuroimaging/data/preprocessed/"
# subjectDir = "/git/ml1030_neuroimaging/data/preprocessed/sub-9001/"
# sessionDir = "ses-1/"
# saveDir = "/git/ml1030_neuroimaging/Patrick"

os.chdir(saveDir)
print("Working Directory: ")
print(os.getcwd())
# %% Function Definitions

#returns # of images in time dimension
def imagesinTimeDim (fullImage):
    iterable = image.iter_img(fullImage)
    return sum(1 for _ in iterable)

def imageDimensions (imagePath):
    img = nib.load(imagePath)
    # What is the shape of the 4D image?
    return img.header.get_data_shape()

def loadImages ():
    #recursively add all mask files to paths list
    paths = list(pathlib.Path(dataDir).glob('**/func/*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'))

    # unmasked: _space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
    # masked: **/func/*_space-MNI152NLin2009cAsym_desc-preproc_bold_masked_(final_resamp_intersected)_bold.nii.gz
    fileNames = paths.copy()

    imageDimList = []

    # #remove path from fileNames leaving us just with the raw filename
    # for i in range(len(fileNames)):
    #     imageDimList.append(imageDimensions(fileNames[i].__str__()))
    #     fileNames[i] = os.path.basename(fileNames[i].name)

    imageDimList.append(Parallel(n_jobs=-1, verbose=100)(delayed(imageDimensions)(fileNames[i].__str__()) for i in range(len(fileNames))))

    #    imageDimList.append(Parallel(n_jobs=2, verbose=100)(delayed(imageDimensions)(fileNames[i].__str__()) for i in range(len(fileNames))))


    #remove path from fileNames leaving us just with the raw filename
    for i in range(len(fileNames)):
        fileNames[i] = os.path.basename(fileNames[i].name)

    #create dataframe from paths and filenames
    imgDF = pd.DataFrame(list(zip(paths, fileNames)), 
                            columns =['path', 'filename']) 
    
    print("Images loaded:", len(imgDF['path'].tolist()))
    imgDF['dimensions'] = imageDimList[0]
    return imgDF

#%%

df = loadImages()
df.to_csv('imageDimensions-unmasked.csv')
df.head()