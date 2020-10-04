# %% [markdown]
# # Metadata Organization
# ## Imports
import pandas as pd 
import numpy as np
import os.path
import glob
import pathlib
import functools
import time
import re

# %% [markdown]
# ## Function to find all the regressor file paths
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

# %% [markdown]
# ## Function to find all the BOLD NII file paths
@timer
def find_paths(relDataFolder, subj, sess, func, patt):
    paths = list(pathlib.Path(relDataFolder).glob(
                        os.path.join(subj, sess, func, patt)
                    )
                )
                        
    return paths

# %% [markdown]
# ## Find all the regressor file paths
regressor_paths = find_paths(relDataFolder='../data/preprocessed',
                            subj='sub-*',
                            sess='ses-*',
                            func='func',
                            patt="*confounds_regressors.tsv")
regressor_paths

# %% [markdown]
# ## Find all the BOLD NII file paths
nii_paths = find_paths(relDataFolder='../data/preprocessed',
                        subj='sub-*',
                        sess='ses-*',
                        func='func',
                        patt="*MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
nii_paths

# %% [markdown]
# ## Read the participants.tsv file to find summaries of the subjects
participant_info_df = pd.read_csv(
        '../data/participants.tsv',
        sep='\t'
    )
participant_info_df

# %% [markdown]
# ## Get a mapping Dataframe of subject and which session is the sleep deprived one
@timer
def map_sleepdep(participant_info):
    df = pd.DataFrame(participant_info.loc[:,['participant_id', 'Sl_cond']])
    df.replace('sub-', '', inplace=True, regex=True)
    return df.rename(columns={'participant_id':'subject', 'Sl_cond':'sleepdep_session'})

sleepdep_map = map_sleepdep(participant_info_df)
sleepdep_map

# %% [markdown]
# ## Get Dataframe of subject, session, task, path
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
        components_list.append([subject, session, task, path.__str__(), 0])
    df = pd.DataFrame(components_list, 
                        columns=['subject', 'session', 'task', 'path', 'sleepdep']
                     )
    return df

components_df = get_bids_components(nii_paths)
components_df

# %% [markdown]
# ## Combine logically sleepdep_map and components_df into 1 dataframe
final_df = components_df.merge(sleepdep_map, how='left')

# %% [markdown]
# ## Response column 'sleepdep' imputed from 'session' 'sleepdep_session'
for i in range(len(final_df)):
    if int(final_df['session'].iloc[i]) == int(final_df['sleepdep_session'].iloc[i]):
        final_df['sleepdep'].iloc[i] = 1
final_df

# %% [markdown]
from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img='../masking/sub-9001-9072_resamp_intersected_mask.nii.gz', standardize=False)

# %% [markdown]


# %% [markdown]
final_df["masked array"] = ""

# %% [markdown]
def gen_voxel_df(filepath):
    f_masked = masker.fit_transform(filepath)
    fmri_masked = pd.DataFrame(np.reshape(f_masked.ravel(), newshape=[1,-1]))
    print(fmri_masked.shape)
    return fmri_masked

# %%
tmp_list = []
for i in range(len(final_df)):
    tmp_list.append(gen_voxel_df(final_df['path'].iloc[i]))

# %%
def merge_df(list):
    return pd.concat(list, ignore_index=True)
tmp_df = merge_df(tmp_list)

# %% [markdown]
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
tmp_df['sleepdep'] = final_df['sleepdep']

# %%
# PyCaret setup
from pycaret.classification import *
from pycaret.clustering import *
clf1 = setup(data=tmp_df,  target='sleepdep')

# %%
# PyCaret compare models, uncomment 'compare_models()' to run
# NOTE: but DON'T DO IT IF LONG RUNTIME!
# compare_models()

# %%
# PyCaret create and run SVM
svm = create_model('svm')