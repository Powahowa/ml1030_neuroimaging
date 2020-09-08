# %%
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd 
import numpy as np

import configurations
configs = configurations.Config('sub-xxx-resamp-intersected')

df = pd.read_pickle(configs.rawVoxelFile)

# %%
# Pipeline: first taking the KBest features and then passing to SVC.
pipeline = Pipeline([
    ('anova', SelectKBest(f_classif, k=50000)),
    ('svc', SVC(kernel='linear'))
])
pipeline.steps

svc = pipeline
# svc_ovo = OneVsOneClassifier(pipeline)
# No need for OneVsRest because it's binary classification
# svc_ovr = OneVsRestClassifier(pipe)

# %% [markdown]
# ### Normalize X
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 
Y = df['sleepdep']
X_array = scaler.fit_transform(df.drop('sleepdep', axis=1))

# %%
from sklearn.model_selection import cross_val_score

cv_scores_svc = cross_val_score(svc, X_array, Y, cv=2, verbose=2, n_jobs=-1)
# cv_scores_ovo = cross_val_score(svc_ovo, X_array, Y, cv=5, verbose=1, n_jobs=1)
# No need for OneVsRest because it's binary classification
# cv_scores_ovr = cross_val_score(svc_ovr, X_array, Y, cv=5, verbose=1, n_jobs=1)

print('SVC Mean CV Score:', cv_scores_svc.mean())
# print('OneVsOne Mean CV Score:', cv_scores_ovo.mean())
# No need for OneVsRest because it's binary classification
# print('OneVsRest Mean CV Score:', cv_scores_ovr.mean())

# %%
# NOTE:
# !!!!!!!!!!!!!!! USING PYCARET BELOW !!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!! ONLY USE BELOW WHEN EVERYTHING IS READY !!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!! ONLY USE BELOW WHEN EVERYTHING IS READY !!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!! ONLY USE BELOW WHEN EVERYTHING IS READY !!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!! ONLY USE BELOW WHEN EVERYTHING IS READY !!!!!!!!!!!!!!!!!!!!!!!

# %%
# Load the df for PyCaret

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