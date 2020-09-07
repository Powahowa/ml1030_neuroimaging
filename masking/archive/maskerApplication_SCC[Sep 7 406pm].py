# %%
import pandas as pd 
import numpy as np
import os.path
import glob
import pathlib
import functools
import time
#from pycaret.classification import *
import re

from nilearn import plotting
from nilearn import image
import nilearn
import nibabel as nib
import numpy as np
from nilearn.input_data import NiftiMasker
from nilearn.image import load_img, math_img

from joblib import Parallel, delayed

# %%
# Function to find all the regressor file paths
def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f'Calling {func.__name__!r}')
        startTime = time.perf_counter()
        value = func(*args, **kwargs)
        endTime = time.perf_counter()
        runTime = endTime - startTime
        print(f'Finished {func.__name__!r} in {runTime:.4f} secs')
        return value
    return wrapper

# %%
# Function to find all the BOLD NII file paths
@timer
def find_paths(relDataFolder, subj, sess, func, patt):
    paths = list(pathlib.Path(relDataFolder).glob(
                        os.path.join(subj, sess, func, patt)
                    )
                )
                        
    return paths

# %%

# #Find all the BOLD NII file paths [LOCAL]
# nii_paths = find_paths(relDataFolder='../data/preprocessed',
#                         subj='sub-*',
#                         sess='ses-*',
#                         func='func',
#                         patt="*MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
# #[LOCAL]
# originalDataPath = "..\\\\data\\\\preprocessed\\\\"
# maskedDataPath = ".\\\\masked_BOLD_images\\\\"
# pathSep = "\\"


#[CAMH SCC]
#Find all the BOLD NII file paths [LOCAL]
nii_paths = find_paths(relDataFolder='/external/rprshnas01/netdata_kcni/edlab/ds000201_preproc/data/derived/fmriprep',
                        subj='sub-*',
                        sess='ses-*',
                        func='func',
                        patt="*MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
nii_paths

#[CAMH SCC]
originalDataPath = "/external/rprshnas01/netdata_kcni/edlab/ds000201_preproc/data/derived/fmriprep/"
maskedDataPath = "/external/rprshnas01/netdata_kcni/edlab/temp_dataknights/masked_BOLD_images/"
pathSep = "/"
# %%
# Prep for next cell
# session_info_df = pd.read_csv(
#         '../data/SleepDiaryData_160320_pseudonymized_final.tsv',
#         sep='\t'
#     )
# session_info_df

# %%
# # Get a mapping Dataframe of subject and which session is the sleep deprived one
# @timer
# def map_sleepdep(session_info):
#     df = pd.DataFrame(session_info.loc[:,['participant_id', 'Sl_cond']])
#     df = df.groupby(['participant_id']).max()
#     return df.rename(columns={'participant_id':'subject', 'Sl_cond':'sleepdep_session'})

# sleepdep_map = map_sleepdep(session_info_df)
# sleepdep_map

# %%
# Get Dataframe of subject, session, task, path
def get_bids_components(paths):
    components_list = []
    for path in paths:
        filename = path.stem
        matches = re.search(
            '[a-z0-9]+\-([a-z0-9]+)_[a-z0-9]+\-([a-z0-9]+)_[a-z0-9]+\-([a-z0-9]+)', 
            filename
        )
        subject = matches.group(1)
        session = matches.group(2)
        task = matches.group(3)
        components_list.append([subject, session, task, path.__str__()])
    df = pd.DataFrame(components_list, 
                        columns=['subject', 'session', 'task', 'path']
                     )
    return df

components_df = get_bids_components(nii_paths)

new_filename_list = []
new_directory_list = []

for i in range(len(components_df)):
    new_filename_list.append(os.path.basename(components_df['path'].iloc[i]))
    new_directory_list.append(os.path.dirname(components_df['path'].iloc[i]))

components_df['new_filename'] = new_filename_list
components_df['new_directory'] = new_directory_list
components_df['new_directory'] = components_df['new_directory'].replace(originalDataPath, maskedDataPath, regex=True)

components_df.to_csv("components_df.csv")

#%% apply mask to all existing BOLD files

def writeAppliedMasks (i):
    cropMask = NiftiMasker(mask_img="./finalMask/final_resamp_intersected_mask.nii.gz", standardize=False)
    cropImage = cropMask.inverse_transform(X=cropMask.fit_transform(image.load_img(components_df.path.iloc[i])))

    os.makedirs([components_df['new_directory'].iloc[i]].__str__()[2:-2], exist_ok=True)
    filename = components_df['new_directory'].iloc[i] + pathSep + components_df['new_filename'].iloc[i][:-7] + "_masked_(final_resamp_intersected)_bold.nii.gz"
    cropImage.to_filename(filename)

#if you run out of memory change n_jobs to the max number of BOLD files you can store in memory
Parallel(n_jobs=-1, verbose=100)(delayed(writeAppliedMasks)(i) for i in range(len(components_df)))


#%% load image back and plot as a test

# testFromDisk1 = image.index_img(components_df.path.iloc[0][:-7] + "_masked_(sub-9001-9072_resamp_intersected)_bold.nii.gz", 0)

# testFromDisk2 = image.index_img(components_df.path.iloc[1][:-7] + "_masked_(sub-9001-9072_resamp_intersected)_bold.nii.gz", 0)

# #using load image
# plt = nilearn.plotting.plot_img(testFromDisk1, cut_coords=[0,0,0], title="Cropped (Mask Applied) Test Image 1 [FROM DISK]")

# #using load image
# plt = nilearn.plotting.plot_img(testFromDisk2, cut_coords=[0,0,0], title="Cropped (Mask Applied) Test Image 2 [FROM DISK]")

#%% apply mask to existing BOLD files [test with one image]

#old inefficient version, next 4 lines
# currentImage = image.load_img(components_df.path.iloc[0])
# cropMask = NiftiMasker(mask_img="final_resamp_intersected_mask.nii.gz", standardize=False)
# maskedArray = cropMask.fit_transform(currentImage)
# cropImage = cropMask.inverse_transform(X=maskedArray)

# cropMask = NiftiMasker(mask_img="final_resamp_intersected_mask.nii.gz", standardize=False)
# cropImage = cropMask.inverse_transform(X=cropMask.fit_transform(image.load_img(components_df.path.iloc[0])))


# #Tony is going to have a heart attack if he sees this. Makes the new directory but it needs a string so we cast it, then we remove the first 2 and last 2 chars to remove the square brackets and quotes
# os.makedirs([components_df['new_directory'].iloc[0]].__str__()[2:-2])
# filename = components_df['new_directory'].iloc[0] + pathSep + components_df['new_filename'].iloc[0][:-7] + "_masked_(final_resamp_intersected)_bold.nii.gz"
# cropImage.to_filename(filename)

# # load image back and plot as a test

# testFromDisk = image.index_img(filename, 0)

# # #using load image
# plt = nilearn.plotting.plot_img(testFromDisk, cut_coords=[0,0,0], title="Cropped (Mask Applied) Test Image [FROM FILE]")

# %%
