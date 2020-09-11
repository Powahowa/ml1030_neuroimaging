# %%
import pandas as pd 
import numpy as np
import os.path
import glob
import pathlib
import functools
import time
from pycaret.classification import *
import matplotlib.pyplot as plt
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

imge = nib.load(fdd)
data = image.get_data(image.index_img(imge, 100))
data

#%%
#print(image.load_img(fdd).shape)
test = np.fft.fftn(data)

#%%
subjectDir = "../data/preprocessed/sub-9001/"
sessionDir = "ses-1/"
slice = image.index_img(subjectDir + 
        sessionDir + 
        "func/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 100)
slice2d = image.get_data(slice)
slice2d
z = slice2d[0:104,123,0:81]

#%%
test = np.fft.fftn(z)

plt.imshow(z.T, cmap='hot', interpolation='nearest', origin = 'lower')
plt.show()

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

#%% 
# Create columns for confounds
# 9 confounds per Dr.Dickie
components_df["csf"] = ""
components_df["white_matter"] = ""
components_df["global_signal"] = ""
components_df["trans_x"] = ""
components_df["trans_y"] = ""
components_df["trans_z"] = ""
components_df["rot_x"] = ""
components_df["rot_y"] = ""
components_df["rot_z"] = ""

# Their first derivatives
components_df["csf_derivative1"] = ""
components_df["white_matter_derivative1"] = ""
components_df["global_signal_derivative1"] = ""
components_df["trans_x_derivative1"] = ""
components_df["trans_y_derivative1"] = ""
components_df["trans_z_derivative1"] = ""
components_df["rot_x_derivative1"] = ""
components_df["rot_y_derivative1"] = ""
components_df["rot_z_derivative1"] = ""

# Square of the confounds
components_df["csf_power2"] = ""
components_df["white_matter_power2"] = ""
components_df["global_signal_power2"] = ""
components_df["trans_x_power2"] = ""
components_df["trans_y_power2"] = ""
components_df["trans_z_power2"] = ""
components_df["rot_x_power2"] = ""
components_df["rot_y_power2"] = ""
components_df["rot_z_power2"] = ""

# Square of the first derivatives
components_df["csf_derivative1_power2"] = ""
components_df["white_matter_derivative1_power2"] = ""
components_df["global_signal_derivative1_power2"] = ""
components_df["trans_x_derivative1_power2"] = ""
components_df["trans_y_derivative1_power2"] = ""
components_df["trans_z_derivative1_power2"] = ""
components_df["rot_x_derivative1_power2"] = ""
components_df["rot_y_derivative1_power2"] = ""
components_df["rot_z_derivative1_power2"] = ""




#%%
# Get Dataframe of subject, session, task, confounds
def get_confounds(paths):
    confounds_list = []
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
    df = pd.DataFrame(confounds_list, 
                        columns=['subject', 'session', 'task', 'confounds']
                     )
    return df

confounds_df = get_bids_components(regressor_paths)
confounds_df

#%%
# Get confounds into components_df 
for r in range(len(confounds_df)):
    x = pd.read_csv(confounds_df['path'].iloc[r].__str__(), sep="\t")
    components_df['csf'].iloc[r] = x['csf'].loc[20:60].values
    components_df['white_matter'].iloc[r] = x['white_matter'].loc[20:60].values
    components_df['global_signal'].iloc[r] = x['global_signal'].loc[20:60].values
    components_df['trans_x'].iloc[r] = x['trans_x'].loc[20:60].values
    components_df['trans_y'].iloc[r] = x['trans_y'].loc[20:60].values
    components_df['trans_z'].iloc[r] = x['trans_z'].loc[20:60].values
    components_df['rot_x'].iloc[r] = x['rot_x'].loc[20:60].values
    components_df['rot_y'].iloc[r] = x['rot_y'].loc[20:60].values
    components_df['rot_z'].iloc[r] = x['rot_z'].loc[20:60].values
    components_df['csf_derivative1'].iloc[r] = x['csf_derivative1'].loc[20:60].values
    components_df['white_matter_derivative1'].iloc[r] = x['white_matter_derivative1'].loc[20:60].values
    components_df['global_signal_derivative1'].iloc[r] = x['global_signal_derivative1'].loc[20:60].values
    components_df['trans_x_derivative1'].iloc[r] = x['trans_x_derivative1'].loc[20:60].values
    components_df['trans_y_derivative1'].iloc[r] = x['trans_y_derivative1'].loc[20:60].values
    components_df['trans_z_derivative1'].iloc[r] = x['trans_z_derivative1'].loc[20:60].values
    components_df['rot_x_derivative1'].iloc[r] = x['rot_x_derivative1'].loc[20:60].values
    components_df['rot_y_derivative1'].iloc[r] = x['rot_y_derivative1'].loc[20:60].values
    components_df['rot_z_derivative1'].iloc[r] = x['rot_z_derivative1'].loc[20:60].values
    components_df['csf_power2'].iloc[r] = x['csf_power2'].loc[20:60].values
    components_df['white_matter_power2'].iloc[r] = x['white_matter_power2'].loc[20:60].values
    components_df['global_signal_power2'].iloc[r] = x['global_signal_power2'].loc[20:60].values
    components_df['trans_x_power2'].iloc[r] = x['trans_x_power2'].loc[20:60].values
    components_df['trans_y_power2'].iloc[r] = x['trans_y_power2'].loc[20:60].values
    components_df['trans_z_power2'].iloc[r] = x['trans_z_power2'].loc[20:60].values
    components_df['rot_x_power2'].iloc[r] = x['rot_x_power2'].loc[20:60].values
    components_df['rot_y_power2'].iloc[r] = x['rot_y_power2'].loc[20:60].values
    components_df['rot_z_power2'].iloc[r] = x['rot_z_power2'].loc[20:60].values
    components_df['csf_derivative1_power2'].iloc[r] = x['csf_derivative1_power2'].loc[20:60].values
    components_df['white_matter_derivative1_power2'].iloc[r] = x['white_matter_derivative1_power2'].loc[20:60].values
    components_df['global_signal_derivative1_power2'].iloc[r] = x['global_signal_derivative1_power2'].loc[20:60].values
    components_df['trans_x_derivative1_power2'].iloc[r] = x['trans_x_derivative1_power2'].loc[20:60].values
    components_df['trans_y_derivative1_power2'].iloc[r] = x['trans_y_derivative1_power2'].loc[20:60].values
    components_df['trans_z_derivative1_power2'].iloc[r] = x['trans_z_derivative1_power2'].loc[20:60].values
    components_df['rot_x_derivative1_power2'].iloc[r] = x['rot_x_derivative1_power2'].loc[20:60].values
    components_df['rot_y_derivative1_power2'].iloc[r] = x['rot_y_derivative1_power2'].loc[20:60].values
    components_df['rot_z_derivative1_power2'].iloc[r] = x['rot_z_derivative1_power2'].loc[20:60].values

    
#%%
# create column for sleep deprived - y/n
components_df["slp_dep"] = ""

#%%
for j in range(len(components_df)):
    if(components_df['subject'].iloc[j])

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
#* https://fmriprep.org/en/latest/outputs.html#confounds
#* https://www.youtube.com/watch?v=sK1GeXDDGD8
#* https://fmriprep.org/en/0.8.1/_modules/fmriprep/workflows/bold/confounds.html
#* https://www.youtube.com/watch?v=qgKm3EayUWY
#* https://www.youtube.com/watch?v=883bzPU6ndU
#* https://www.youtube.com/watch?v=gMuxB18Cr0o
#* https://neurostars.org/


