# %%
import pandas as pd 
import numpy as np
import os.path
import glob
import pathlib
import functools
import time
import matplotlib.pyplot as plt
import matplotlib
import re
from nilearn import image
import nibabel as nib
from joblib import Parallel, delayed
import gc

# %% [markdown]
# ## Load configs (all patterns/files/folderpaths)
import configurations
configs = configurations.Config('patrickTest')

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
                        patt=configs.maskedImagePattern)
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
    for i, path in enumerate(paths):
        filename = path.stem
        dirpath = path.parents[0]
        matches = re.search(
            '[a-z0-9]+\-([a-z0-9]+)_[a-z0-9]+\-([a-z0-9]+)_[a-z0-9]+\-([a-z0-9]+)', 
            filename
        )
        subject = matches.group(1)
        session = matches.group(2)
        task = matches.group(3)
        confound_file = path.with_name(
            'sub-'+subject+'_ses-'+session+'_task-'+task+'_desc-confounds_regressors.tsv'
        )
        components_list.append([subject, session, task, 
            path.__str__(), confound_file.__str__(), 0]
        )
    df = pd.DataFrame(components_list, 
        columns=['subject', 'session', 'task', 'path', 'confound_path', 'sleepdep']
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
# ## Get confounds that can be used further clean up the signal or for prediction
def get_important_confounds(regressor_paths, important_reg_list, start, end):
    regressors_df_list = []
    for paths in regressor_paths:
        regressors_all = pd.DataFrame(pd.read_csv(paths, sep="\t"))
        regressors_selected = pd.DataFrame(regressors_all[important_reg_list].loc[start:end-1])
        regressors_df_list.append(pd.DataFrame(regressors_selected.stack(0)).transpose())
    concatenated_df = pd.concat(regressors_df_list, ignore_index=True)
    concatenated_df.columns = [col[1] + '-' + str(col[0]) for col in concatenated_df.columns.values]
    return concatenated_df

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

important_confounds_df = get_important_confounds(
    sleep_bids_comb_df['confound_path'], important_reg_list, configs.startSlice, configs.endSlice
)

# %% [markdown]
# ## Load the masker data file to prepare to apply to images

#masker = NiftiMasker(mask_img=configs.maskDataFile, standardize=False)

# %% [markdown]
# ## Helper to generate raw voxel df from a given path + masker and print shape for sanity
@timer
def genFFT(filepath, start, end):
    FFTarray = np.empty(0)
    for i in range(start, end):
        FFT = np.fft.fftn(image.get_data(image.index_img(filepath, slice(i))))
        FFTarray = np.concatenate((FFTarray, FFT), axis=0)
    reshaped_array = np.ravel(FFTarray)
    return reshaped_array

# %% [markdown]
# ## Function to generate from masked image the raw voxel df from all images in folder
@timer
def get_FFT_df(metadata_df, start, end):

    feature_array = genFFT(metadata_df['path'].iloc[0], start, end)
    print("feature_array at index 0")
    print(feature_array)
    print()
    feature_array = np.vstack((feature_array, np.vstack(Parallel(n_jobs=-1, verbose=100)(delayed(genFFT)(metadata_df['path'].iloc[i], masker, start, end) for i in range(1, len(metadata_df))))))

    tmp_df = pd.DataFrame(feature_array)
    print() # Print to add a spacer for aesthetics
    tmp_df['sleepdep'] = metadata_df['sleepdep']
    temp_dict = dict((val, str(val)) for val in list(range(len(tmp_df.columns)-1)))
    return tmp_df.rename(columns=temp_dict, errors='raise')

# %% [markdown]
# ## Garbage collect
gc.collect()

# %% [markdown]
# ## Get/Generate raw voxels dataframe from all images with Y column label included
FFT_df = get_FFT_df(sleep_bids_comb_df, configs.startSlice, configs.endSlice)
#X = pd.concat([voxels_df, important_confounds_df], axis=1)
X = FFT_df

# %% [markdown]
# ## Separately get the Y label
Y = sleep_bids_comb_df['sleepdep']

# %% [markdown]
# ## Save raw dataframe with Y column included to a file
X.to_pickle(configs.rawVoxelFile)



#%%

#FFT test

fdd = '../data/preprocessed/sub-9001/ses-1/func/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'

FFT = np.fft.fftn(image.get_data(image.index_img(fdd, slice(50))))



#%%

testImg = nib.load(fdd)
data = image.get_data(image.index_img(testImg, 100))
data

#%%
#print(image.load_img(fdd).shape)
test = np.fft.fftn(data)

#%%
subjectDir = "../data/preprocessed/sub-9001/"
sessionDir = "ses-1/"
slice3d = image.index_img(subjectDir + 
        sessionDir + 
        "func/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 100)
slice3d = image.get_data(slice3d)
slice3d = slice3d[52,0:123,0:81]

#%%

plt.imshow(slice3d.T, cmap='hot', interpolation='nearest', origin = 'lower')
plt.show()

#%%
testFFT = np.fft.fftn(slice3d)
testFFTFloat = testFFT.real
#plt.imshow(testFFTFloat.T, cmap='hot', interpolation='nearest', origin ='lower', norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))
plt.plot(np.log10(testFFT))
plt.savefig('FFT of 2D slice.png', dpi=400)