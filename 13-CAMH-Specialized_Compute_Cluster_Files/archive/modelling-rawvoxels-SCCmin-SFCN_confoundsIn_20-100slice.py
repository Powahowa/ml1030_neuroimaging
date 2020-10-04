# %%
# ## Imports
import pandas as pd
import numpy as np
# import tensorflow as tf
# from tensorflow import keras
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
        GradientBoostingClassifier, StackingClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import balanced_accuracy_score, accuracy_score, \
        precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import cross_validate

#plt.style.use('ggplot')
import matplotlib as plt
plt.use('Agg')

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import configurations
configs = configurations.Config('SFCN_confoundsIn_20-100slice')

X = pd.read_pickle(configs.rawVoxelFile)

# %%
# Pipeline: first taking the KBest features and then passing to SVC.
pipeline = Pipeline([
    ('anova', SelectKBest(f_classif, k=50000)),
    ('svc', SVC(kernel='linear'))
])
pipeline.steps

svc = pipeline

# %% [markdown]
# ### Normalize X
scaler = MinMaxScaler() 
y = X['sleepdep']
X_array = scaler.fit_transform(X.drop('sleepdep', axis=1))
df2 = pd.DataFrame(X_array)
df2['sleepdep'] = X['sleepdep']

# %%
from sklearn.model_selection import cross_val_score

cv_scores_svc = cross_val_score(svc, X_array, y, cv=6, verbose=100, n_jobs=-1)

print('SVC Mean CV Score:', cv_scores_svc.mean())