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

from pycaret.clustering import *

# %%
# Load the df for PyCaret
df = pd.read_pickle('rawvoxelsdf.pkl')

# # %%
# # PyCaret clustering setup

# # ### import clustering module
# from pycaret.clustering import *

# # ### intialize the setup
# clf1 = setup(data=df)

# # %%
# # ### create k-means model
# kmeans = create_model('kmeans')

# # %%
# # PyCaret compare models, uncomment 'compare_models()' to run
# # NOTE: but DON'T DO IT IF LONG RUNTIME!
# # compare_models()

# %%
# PyCaret create and run SVM
from pycaret.clustering import *
df.drop('sleepdep', axis=1)
clf1 = setup(data=df, silent=True)
kmeans = create_model('kmeans')
plot_model(kmeans, save=True)
# %%
logs = get_logs(save=True)