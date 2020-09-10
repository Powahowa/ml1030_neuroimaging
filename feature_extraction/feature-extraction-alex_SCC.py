# %%
# import statements
# import datasets
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
from nilearn.input_data import NiftiLabelsMasker

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import functools
import time
import pathlib
import os.path
import re

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
# # ## Get confounds that can be used further clean up the signal or for prediction
# def get_important_confounds(regressor_paths, important_reg_list, start, end):
#     regressors_df_list = []
#     for paths in regressor_paths:
#         regressors_all = pd.DataFrame(pd.read_csv(paths, sep="\t"))
#         regressors_selected = pd.DataFrame(regressors_all[important_reg_list].loc[start:end-1])
#         regressors_df_list.append(pd.DataFrame(regressors_selected.stack(0)).transpose())
#     concatenated_df = pd.concat(regressors_df_list, ignore_index=True)
#     concatenated_df.columns = [col[1] + '-' + str(col[0]) for col in concatenated_df.columns.values]
#     return concatenated_df

# important_reg_list = ['csf', 'white_matter', 'global_signal', 
#                       'trans_x', 'trans_y', 'trans_z', 
#                       'rot_x', 'rot_y', 'rot_z', 
#                       'csf_derivative1', 'white_matter_derivative1', 'global_signal_derivative1',
#                       'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
#                       'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
#                       'csf_power2', 'white_matter_power2', 'global_signal_power2',
#                       'trans_x_power2', 'trans_y_power2', 'trans_z_power2',
#                       'rot_x_power2', 'rot_y_power2', 'rot_z_power2',
#                       'csf_derivative1_power2', 'white_matter_derivative1_power2', 'global_signal_derivative1_power2',
#                       'trans_x_derivative1_power2', 'trans_y_derivative1_power2', 'trans_z_derivative1_power2',
#                       'rot_x_derivative1_power2', 'rot_y_derivative1_power2', 'rot_z_derivative1_power2'
#                      ]

# important_confounds_df = get_important_confounds(
#     sleep_bids_comb_df['confound_path'], important_reg_list, configs.startSlice, configs.endSlice
# )

# %%
# fetch Harvard Oxford atlas
dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_filename = dataset.maps
labels = dataset.labels


# %%

time_series_list = []
masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=False,
                           memory='nilearn_cache', verbose=5)

#masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True, memory='nilearn_cache', memory_level=1, verbose=0)

# Here we go from nifti files to the signal time series in a numpy
# array. Note how we give confounds to be regressed out during signal
# extraction

#time_series = masker.fit_transform(arrows_slices_file_path, confounds=no_nan_arrows_confounds_regressors_path)
#time_series = masker.fit_transform(arrows_slices_file_path, confounds=zero_for_nan_arrows_confounds_regressors_path)

for nii_path in nii_paths:
    time_series = masker.fit_transform(nii_path.__str__())
    time_series_list.append(time_series)

# %%
# Calculate classification for connectivity

# for now, we just use "correlation"
# kinds_of_matrix_correlation = ['correlation', 'partial correlation', 'tangent']
kinds_of_matrix_correlation = ['correlation', 'partial correlation', 'tangent']

# classes: "0" is ses-1, "1" is ses-2
classes = sleep_bids_comb_df['sleepdep'].tolist()
#classes = take binary column from dataframe and make it into a list

# define cross-validation strategy here
cv = StratifiedShuffleSplit(n_splits=24, random_state=0, test_size=5)

# convert list into numpy array
time_series_numpy_array = np.asarray(time_series_list)

# scores
scores = {}

# %%

def createConnectivityMeasure(train, test):
    scores[kind_of_matrix_correlation] = []
    # vectorize turns it into a 1D array
    connectivity = ConnectivityMeasure(kind=kind_of_matrix_correlation, vectorize=True)
    
    #calculate vectorized connectome for training set
    connectomes = connectivity.fit_transform(time_series_numpy_array[train])
    #print(len(connectomes))
    #print(len(classes[train]))
    
    classes_np_array = np.array(classes)
    
    #fit classifier
    classifier = LinearSVC().fit(connectomes, classes_np_array[train])
    
    predictions = classifier.predict(
        connectivity.transform(time_series_numpy_array[test]))
    
    # store the accuracy for this cross-validation fold
    #scores[kind_of_matrix_correlation].append(accuracy_score(classes_np_array[test], predictions))
    return accuracy_score(classes_np_array[test], predictions)

for kind_of_matrix_correlation in kinds_of_matrix_correlation:
    scores[kind_of_matrix_correlation] = []
    score = Parallel(n_jobs=-1, verbose=50)(delayed(createConnectivityMeasure)(train, test) for train, test in cv.split(time_series_numpy_array, classes))
    scores[kind_of_matrix_correlation].append(score)
  
# calculate mean accuracy scores, and their standard deviations
mean_scores = [np.mean(scores[kind_of_matrix_correlation]) for kind_of_matrix_correlation in kinds_of_matrix_correlation]
scores_std = [np.std(scores[kind_of_matrix_correlation]) for kind_of_matrix_correlation in kinds_of_matrix_correlation]

# output results into a df
results_df = pd.DataFrame(list(zip(kinds_of_matrix_correlation, mean_scores, scores_std)), columns = ['Kind of correlation', 'mean_scores', 'scores_std'])

# print results of df to a csv
results_df.to_csv('test_classification_of_functional_connectivity_between_roi_results.csv')
# %%
