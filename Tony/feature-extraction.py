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
import gc
from nilearn.input_data import NiftiMasker

# %% [markdown]
# ## Load configs
import configurations
configs = configurations.Config('default')

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
regressor_paths = find_paths(relDataFolder=configs.dataDir,
                            subj='sub-*',
                            sess='ses-*',
                            func='func',
                            patt=configs.confoundsFilePattern)
regressor_paths

# %% [markdown]
# ## Find all the BOLD NII file paths
nii_paths = find_paths(relDataFolder=configs.dataDir,
                        subj='sub-*',
                        sess='ses-*',
                        func='func',
                        patt=configs.preprocessedImagePattern)
nii_paths

# %% [markdown]
# ## Read the participants.tsv file to find summaries of the subjects
participant_info_df = pd.read_csv(
        configs.participantsSummaryFile,
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
@timer
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

bids_comp_df = get_bids_components(nii_paths)
bids_comp_df

# %% [markdown]
# ## Combine logically sleepdep_map and components_df into 1 dataframe
sleep_bids_comb_df = bids_comp_df.merge(sleepdep_map, how='left')

# %% [markdown]
# ## Response column 'sleepdep' imputed from 'session' 'sleepdep_session'
for i in range(len(sleep_bids_comb_df)):
    if (int(sleep_bids_comb_df['session'].iloc[i]) == 
            int(sleep_bids_comb_df['sleepdep_session'].iloc[i])):
        sleep_bids_comb_df['sleepdep'].iloc[i] = 1
sleep_bids_comb_df

# %% [markdown]
masker = NiftiMasker(
    mask_img=configs.maskDataFile, 
    standardize=False
    )

# %% [markdown]
# sleep_bids_comb_df["masked array"] = ""

# %% [markdown]
@timer
def gen_one_masked_df(filepath, masker):
    file_masked = masker.fit_transform(filepath)
    fmri_masked = pd.DataFrame(np.reshape(
        file_masked.ravel(), newshape=[1,-1]), dtype='float32')
    print('Masked shape of raw voxels for file \"' +
          str(pathlib.Path(filepath).stem) + 
          '\" is: ' + 
          str(fmri_masked.shape)) 
    return fmri_masked

# %%
@timer
def get_voxels_df(metadata_df, masker):
    rawvoxels_list = []
    for i in range(len(metadata_df)-18):
        rawvoxels_list.append(gen_one_masked_df(metadata_df['path'].iloc[i], masker))
    tmp_df = pd.concat(rawvoxels_list, ignore_index=True)
    tmp_df['sleepdep'] = metadata_df['sleepdep']
    temp_dict = dict((val, str(val)) for val in list(range(len(tmp_df.columns)-1)))
    return tmp_df.rename(columns=temp_dict, errors='raise')

# %%
gc.collect()

# %%
X = get_voxels_df(sleep_bids_comb_df, masker)

# %%
Y = sleep_bids_comb_df['sleepdep']

# %%
# import pyarrow.feather as feather
# feather.write_feather(X, './rawvoxelsdf.feather')

# %%
X.to_pickle(configs.rawVoxelFile)

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
# Load the df for PyCaret
df = pd.read_pickle(configs.rawVoxelFile)

# %%
# PyCaret clustering setup

# ### import clustering module
from pycaret.clustering import *

# ### intialize the setup
clf1 = setup(data=df)

# %%
# ### create k-means model
kmeans = create_model('kmeans')

# %%
# PyCaret compare models, uncomment 'compare_models()' to run
# NOTE: but DON'T DO IT IF LONG RUNTIME!
# compare_models()

# %%
# PyCaret create and run SVM
from pycaret.classification import *
clf1 = setup(data=df,  target='sleepdep')
svm = create_model('svm')
# %%
