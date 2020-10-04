# %%
import pandas as pd 
import numpy as np
import os.path
import glob
import pathlib
import functools
import time
from pycaret.classification import *
import re

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
# Find all the regressor file paths
regressor_paths = find_paths(relDataFolder='../data/preprocessed',
                            subj='sub-*',
                            sess='ses-*',
                            func='func',
                            patt="*confounds_regressors.tsv")
regressor_paths

# %%
# Find all the BOLD NII file paths
nii_paths = find_paths(relDataFolder='../data/preprocessed',
                        subj='sub-*',
                        sess='ses-*',
                        func='func',
                        patt="*MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
nii_paths

# %%
# Prep for next cell
session_info_df = pd.read_csv(
        '../data/SleepDiaryData_160320_pseudonymized_final.tsv',
        sep='\t'
    )
session_info_df

# %%
# Get a mapping Dataframe of subject and which session is the sleep deprived one
@timer
def map_sleepdep(session_info):
    df = pd.DataFrame(session_info.loc[:,['participant_id', 'Sl_cond']])
    df = df.groupby(['participant_id']).max()
    return df.rename(columns={'participant_id':'subject', 'Sl_cond':'sleepdep_session'})

sleepdep_map = map_sleepdep(session_info_df)
sleepdep_map

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
        components_list.append([subject, session, task, path])
    df = pd.DataFrame(components_list, 
                        columns=['subject', 'session', 'task', 'path']
                     )
    return df

components_df = get_bids_components(nii_paths)
components_df

#%% plot a 4D image
fdd = '../data/preprocessed/sub-9001/ses-1/func/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'

from nilearn import image
import nibabel as nib

print(image.load_img(fdd).shape)

#%% 

#%% Feature Generation with NiftiMasker
from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img='../masking/sub-9001-9072_resamp_intersected_mask.nii.gz', standardize=True)

#%% 
fmri_masked = []

#%%
components_df["masked array"] = ""

#%%
for i in range(len(components_df)):
    f_masked = masker.fit_transform(components_df['path'].iloc[i].__str__())
    #fmri_masked.append(f_masked)
    components_df['masked array'].iloc[i] = f_masked

# %%
# TODO: Combine logically sleepdep_map and components_df into 1 dataframe


# %%
# Regressors only to be used to further clean up the signal
# Will need to run GLM through the images or something
@timer
def gen_common_regressors(regressor_paths):
    df_list = []
    for paths in regressor_paths:
        temp = pd.DataFrame(pd.read_csv(paths, sep='\t'))
        df_list.append(temp)
    return pd.concat(df_list, join='inner', ignore_index=True)

common_regressors = gen_common_regressors(regressor_paths)
common_regressors

# %%
# NOTE:
# !!!!!!!!!!!!!!! ONLY USE BELOW WHEN EVERYTHING IS READY !!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!! ONLY USE BELOW WHEN EVERYTHING IS READY !!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!! ONLY USE BELOW WHEN EVERYTHING IS READY !!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!! ONLY USE BELOW WHEN EVERYTHING IS READY !!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!! ONLY USE BELOW WHEN EVERYTHING IS READY !!!!!!!!!!!!!!!!!!!!!!!

# %%
# PyCaret setup
clf1 = setup(data=df,  target='Sleep_Deprived')

# %%
# PyCaret compare models, uncomment 'compare_models()' to run
# NOTE: but DON'T DO IT IF LONG RUNTIME!
# compare_models()

# %%
# PyCaret create and run SVM
svm = create_model('svm')


#%% [markdown]
# References
#* https://nilearn.github.io/auto_examples/plot_decoding_tutorial.html