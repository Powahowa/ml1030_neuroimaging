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

import matplotlib as plt
plt.use('Agg')#needed because we don't have an X window system on the SCC

from joblib import Parallel, delayed


# %% [markdown]
# ## Function to find all the BOLD NII file paths

def find_paths(relDataFolder, subj, sess, func, patt):
    paths = list(pathlib.Path(relDataFolder).glob(
                        os.path.join(subj, sess, func, patt)
                    )
                )
                        
    return paths

# %% [markdown]
# ## Find all the regressor file paths
#[SCC]: ../ds000201_preproc/data/derived/fmriprep

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
        'participants.tsv',
        sep='\t'
    )
participant_info_df

# %% [markdown]
# ## Get a mapping Dataframe of subject and which session is the sleep deprived one

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
    mask_img='final_resamp_intersected_mask.nii.gz', 
    standardize=False
    )

# %% [markdown]
# sleep_bids_comb_df["masked array"] = ""

# %% [markdown]

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

def get_voxels_df(metadata_df, masker):
    rawvoxels_list = []

    #this does work but it's single-threaded
    for i in range(20):
        rawvoxels_list.append(gen_one_masked_df(metadata_df['path'].iloc[i], masker))

    #this doesn't work - I think because it gets CRAZY big and list aren't meant for that
    #rawvoxels_list.append(Parallel(n_jobs=-1, verbose=100)(delayed(gen_one_masked_df)(metadata_df['path'].iloc[i], masker) for i in range(2)))
    
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
X.to_pickle('./rawvoxelsdf-parallel.pkl')

# %% [markdown]
# Regressors only to be used to further clean up the signal
# Will need to run GLM through the images or something

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

