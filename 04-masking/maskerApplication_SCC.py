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
# Function to find all the BOLD NII file paths

def find_paths(relDataFolder, subj, sess, func, patt):
    paths = list(pathlib.Path(relDataFolder).glob(
                        os.path.join(subj, sess, func, patt)
                    )
                )
                        
    return paths

# %%

important_reg_list = ['csf', 'white_matter', 'global_signal', 
                      'trans_x', 'trans_y', 'trans_z', 
                      'rot_x', 'rot_y', 'rot_z', 
                      'csf_derivative1', 'white_matter_derivative1', 'global_signal_derivative1',
                      'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
                      'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
                      'csf_power2', 'white_matter_power2', 'global_signal_power2',
                      'trans_x_power2', 'trans_y_power2', 'trans_z_power2',
                      'rot_x_power2', 'rot_y_power2', 'rot_z_power2',
                      'csf_derivative1_power2', 'white_matter_derivative1_power2', 'global_signal_derivative1_power2',
                      'trans_x_derivative1_power2', 'trans_y_derivative1_power2', 'trans_z_derivative1_power2',
                      'rot_x_derivative1_power2', 'rot_y_derivative1_power2', 'rot_z_derivative1_power2'
                     ]


# #Find all the BOLD NII file paths [LOCAL]
# nii_paths = find_paths(relDataFolder='../data/preprocessed',
#                         subj='sub-*',
#                         sess='ses-*',
#                         func='func',
#                         patt="*MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
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
originalDataPath = "/external/rprshnas01/netdata_kcni/edlab/ds000201_preproc/data/derived/fmriprep/"
maskedDataPath = "/external/rprshnas01/netdata_kcni/edlab/temp_dataknights/masked_BOLD_images/stand=False_confounds=Included/"
pathSep = "/"


nii_paths

# %% find confounds

# #[LOCAL]
# confound_paths = find_paths(relDataFolder='../data/preprocessed',
#                         subj='sub-*',
#                         sess='ses-*',
#                         func='func',
#                         patt="*confounds_regressors.tsv")

#[CAMH SCC]
confound_paths = find_paths(relDataFolder='/external/rprshnas01/netdata_kcni/edlab/ds000201_preproc/data/derived/fmriprep',
                        subj='sub-*',
                        sess='ses-*',
                        func='func',
                        patt="*confounds_regressors.tsv")


confound_paths
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
    confounds_path_list = []
    for i in range(len(df)):
        confounds_path_list.append(df['path'].iloc[i].__str__()[:-50] + "desc-confounds_regressors.tsv")
    df['confounds_path'] = confounds_path_list
    return df

components_df = get_bids_components(nii_paths)

#confounds_df = get_bids_components(confound_paths)
#confounds_df.rename(columns={'path':'confounds_path'}, inplace=True)
#components_df = components_df.join(confounds_df['confounds_path'])

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

    print("Working on:", components_df['path'].iloc[i])

    #read all confounds
    confounds_all = pd.DataFrame(pd.read_csv(components_df['confounds_path'].iloc[i], sep="\\t"))
    #filter confounds to 36 selected confounds
    confounds_selected = np.array(confounds_all[important_reg_list])

    #choose ONE of following statements
    #replace nan with 0
    confounds_selected = np.nan_to_num(confounds_selected)
    #drop nan rows [does not work because then the confound length does not match the image]
    #confounds_selected = confounds_selected[~np.isnan(confounds_selected).any(axis=1)]
    
    cropMask = NiftiMasker(mask_img="./finalMask/final_resamp_intersected_mask_v2.nii.gz", standardize=False)
    cropImage = cropMask.inverse_transform(X=cropMask.fit_transform(image.load_img(components_df['path'].iloc[i]), confounds=confounds_selected))

    os.makedirs([components_df['new_directory'].iloc[i]].__str__()[2:-2], exist_ok=True)
    filename = components_df['new_directory'].iloc[i] + pathSep + components_df['new_filename'].iloc[i][:-7] + "_masked_(final_resamp_intersected_v2)_bold.nii.gz"
    cropImage.to_filename(filename)

#if you run out of memory change n_jobs to the max number of BOLD files you can store in memory
Parallel(n_jobs=1, verbose=100)(delayed(writeAppliedMasks)(i) for i in range(len(components_df)))


#%% load image back and plot as a test

# testFromDisk1 = image.index_img(components_df.path.iloc[0][:-7] + "_masked_(sub-9001-9072_resamp_intersected)_bold.nii.gz", 0)

# testFromDisk2 = image.index_img(components_df.path.iloc[1][:-7] + "_masked_(sub-9001-9072_resamp_intersected)_bold.nii.gz", 0)

# #using load image
# plt = nilearn.plotting.plot_img(testFromDisk1, cut_coords=[0,0,0], title="Cropped (Mask Applied) Test Image 1 [FROM DISK]")

# #using load image
# plt = nilearn.plotting.plot_img(testFromDisk2, cut_coords=[0,0,0], title="Cropped (Mask Applied) Test Image 2 [FROM DISK]")

#%% apply mask to existing BOLD files [test with one image]

# #old inefficient version, next 4 lines
# # currentImage = image.load_img(components_df.path.iloc[0])
# # cropMask = NiftiMasker(mask_img="final_resamp_intersected_mask_v2.nii.gz", standardize=False)
# # maskedArray = cropMask.transform(currentImage)
# # cropImage = cropMask.inverse_transform(X=maskedArray)

# #read all confounds
# confounds_all = pd.DataFrame(pd.read_csv(components_df['confounds_path'].iloc[0], sep="\t"))
# #filter confounds to 36 selected confounds
# confounds_selected = np.array(confounds_all[important_reg_list])

# #choose ONE of following statements
# #replace nan with 0
# confounds_selected = np.nan_to_num(confounds_selected)
# #drop nan rows [does not work because then the confound length does not match the image]
# #confounds_selected = confounds_selected[~np.isnan(confounds_selected).any(axis=1)]

# cropMask = NiftiMasker(mask_img="final_resamp_intersected_mask_v2.nii.gz", standardize=False)
# cropImage = cropMask.inverse_transform(X=cropMask.fit_transform(image.load_img(components_df['path'].iloc[0]), confounds=confounds_selected))

# #Tony is going to have a heart attack if he sees this. Makes the new directory but it needs a string so we cast it, then we remove the first 2 and last 2 chars to remove the square brackets and quotes
# os.makedirs([components_df['new_directory'].iloc[0]].__str__()[2:-2])
# filename = components_df['new_directory'].iloc[0] + pathSep + components_df['new_filename'].iloc[0][:-7] + "_masked_(final_resamp_intersected_v2)_bold.nii.gz"
# cropImage.to_filename(filename)

# # load image back and plot as a test

# testFromDisk = image.index_img(filename, 0)

# # #using load image
# plt = nilearn.plotting.plot_img(testFromDisk, cut_coords=[0,0,0], title="Cropped (Mask Applied) Test Image [FROM FILE]")

# %%
