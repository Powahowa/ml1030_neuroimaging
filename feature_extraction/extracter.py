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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import functools
import time
import pathlib
import os.path

# %%
# Fetch Harvard Oxford Atlas
dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_filename = dataset.maps
labels = dataset.labels

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
# Function to find all file paths
@timer
def find_paths(relDataFolder, subj, sess, func, patt):
    paths = list(pathlib.Path(relDataFolder).glob(
                        os.path.join(subj, sess, func, patt)
                    )
                )
                        
    return paths

# %%
# fetch Harvard Oxford atlas
dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_filename = dataset.maps
labels = dataset.labels

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
time_series_list = []
groups = [] # session 1 or session 2 

# %%
masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
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
    
    if "ses-1" in nii_path.__str__():
        groups.append("ses-1")
    else:
        groups.append("ses-2")

# %%
# Calculate classification for connectivity

# for now, we just use "correlation"
# kinds_of_matrix_correlation = ['correlation', 'partial correlation', 'tangent']
kinds_of_matrix_correlation = ['correlation', 'partial correlation', 'tangent']

# classes: "0" is ses-1, "1" is ses-2
_, classes = np.unique(groups, return_inverse=True)

# define cross-validation strategy here
cv = StratifiedShuffleSplit(n_splits=15, random_state=0, test_size=5)

# convert list into numpy array
time_series_numpy_array = np.asarray(time_series_list)

# scores
scores = {}

# %%
for kind_of_matrix_correlation in kinds_of_matrix_correlation:
    scores[kind_of_matrix_correlation] = []
    
    for train, test in cv.split(time_series_numpy_array, classes):
        # vectorize turns it into a 1D array
        connectivity = ConnectivityMeasure(kind=kind_of_matrix_correlation, vectorize=True)
        
        #calculate vectorized connectome for training set
        connectomes = connectivity.fit_transform(time_series_numpy_array[train])
        #print(len(connectomes))
        #print(len(classes[train]))
        
        #fit classifier
        classifier = LinearSVC().fit(connectomes, classes[train])
        
        predictions = classifier.predict(
            connectivity.transform(time_series_numpy_array[test]))
        
        # store the accuracy for this cross-validation fold
        scores[kind_of_matrix_correlation].append(accuracy_score(classes[test], predictions))

# calculate mean accuracy scores, and their standard deviations
mean_scores = [np.mean(scores[kind_of_matrix_correlation]) for kind_of_matrix_correlation in kinds_of_matrix_correlation]
scores_std = [np.std(scores[kind_of_matrix_correlation]) for kind_of_matrix_correlation in kinds_of_matrix_correlation]

# output results into a df
results_df = pd.DataFrame(list(zip(kinds_of_matrix_correlation, mean_scores, scores_std)), columns = ['Kind of correlation', 'mean_scores', 'scores_std'])

# print results of df to a csv
results_df.to_csv('test_classification_of_functional_connectivity_between_roi_results.csv')