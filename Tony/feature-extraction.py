# %%
import pandas as pd 
import numpy as np
import os.path
import glob
import pathlib
import functools
import time
from pycaret.classification import *

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
@timer
def find_paths(relDataFolder, subj, sess, func, patt):
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
df = pd.DataFrame(pd.read_csv(regressor_paths[0], sep='\t'))
for paths in regressor_paths[1:]:
    temp = pd.DataFrame(pd.read_csv(paths, sep='\t'))
    df.append(temp)
df

# %%
setup(data=df,  target='Legendary')