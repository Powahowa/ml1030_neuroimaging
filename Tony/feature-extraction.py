# %%
import pandas as pd 
import numpy as np
import os.path
import glob
import pathlib
import functools
import time

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
# @timer
# def find_nii(relDataFolder):
#     paths = list(pathlib.Path(relDataFolder).glob(
#                     os.path.join('sub-*', 'ses-*', 'func', 
#                         '*MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')))
                        
#     return paths

# nii_paths = find_nii(pathlib.Path('../../data/preprocessed'))

# %%
@timer
def find_paths(relDataFolder, subj, sess, func, patt):
    # print(pathlib.Path.joinpath(subPath, pattern))
    paths = list(pathlib.Path(relDataFolder).glob(
                    os.path.join(subj, sess, func, patt)))
                        
    return paths

# %%
regressor_paths = find_paths(relDataFolder='../../data/preprocessed',
                            subj='sub-*',
                            sess='ses-*',
                            func='func',
                            patt="*confounds_regressors.tsv")
regressor_paths

# %%
nii_paths = find_paths(relDataFolder='../../data/preprocessed',
                        subj='sub-*',
                        sess='ses-*',
                        func='func',
                        patt="*MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
nii_paths

# %%