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
        precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate
from mlxtend.plotting import plot_learning_curves
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.model_selection import train_test_split

import configurations
configs = configurations.Config('STCM_confoundsOut_43-103slice')

df = pd.read_pickle(configs.rawVoxelFile)

# %% [markdown]
# ### Normalize X
#scaler = MinMaxScaler() 
y = np.array(df['sleepdep']).ravel()
# y = ["WideAwake" if x == 0 else "Sleepy" for x in y['sleepdep']]
X = pd.DataFrame(df.drop('sleepdep', axis=1))
#X = pd.DataFrame(scaler.fit_transform(X))


# %% [markdown]
# ## Try traditional ML models
# ### Define and train the models
models = [  
    LogisticRegression(random_state=1),
    KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
    DecisionTreeClassifier(),
    GaussianNB(),
    LinearSVC(),
    # BaggingClassifier(base_estimator=\
    #     DecisionTreeClassifier(max_leaf_nodes=2620), n_estimators=100, n_jobs=-1)
]
model_namelist = ['Logistic Regression',
                    'KNeighbors',
                  'Decision Tree',
                  'GaussianNB', 
                  'SVM/Linear SVC',
                #   'Bagging-DT'
                  ]
scoring = {'precision': make_scorer(precision_score, average='binary'), 
           'recall': make_scorer(recall_score, pos_label=1, average='binary'), 
           'accuracy': make_scorer(accuracy_score), 
           'f1': make_scorer(f1_score, average='binary'),
           'roc_auc': make_scorer(roc_auc_score)
           # 'mcc': make_scorer(matthews_corrcoef)
          } 

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, \
    test_size=0.20, random_state=0)
# X_train = pd.DataFrame(scaler.fit_transform(X_train))
# X_test = pd.DataFrame(scaler.fit_transform(X_test))

# %%
# ### Loop cross validation through various models and generate results\
cv_result_entries = []
i = 0 
for mod in models:
    metrics = cross_validate(
        mod,
        X_train,
        y_train,
        cv=5,
        scoring = scoring,
        return_train_score=False,
        n_jobs=-1
    )
    for key in metrics.keys():
        for fold_index, score in enumerate(metrics[key]):
            cv_result_entries.append((model_namelist[i], fold_index, key, score))
    i += 1
cv_results_df = pd.DataFrame(cv_result_entries)
cv_results_df.columns = ['algo', 'cv fold', 'metric', 'value']
cv_results_df.to_csv('rawvoxels_cv_results_df.csv')

# %% [markdown]
# ### Plot cv results
for metric_name, metric in zip(['fit_time',
                                'test_precision',
                                'test_recall',
                                'test_f1',
                                'test_roc_auc'],
                                ['Fit Time',
                                'Precision',
                                'Recall',
                                'F1 Score',
                                'ROC AUC']):
    sns.boxplot(x='algo', y='value', #hue='algo',
        data=cv_results_df[cv_results_df.metric.eq(f'{metric_name}')])
    sns.stripplot(x='algo', y = 'value', 
        data = cv_results_df[cv_results_df.metric.eq(f'{metric_name}')],
        size = 5, linewidth = 1)
    plt.title(f'{metric} Algo Comparison', fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel(f'{metric}', fontsize=12)
    plt.xticks([0, 1, 2, 3, 4])
    plt.xticks(rotation=45)
    ## plt.show()
    plt.savefig(configs.saveDir + 'rawVoxels_cv_results.png')


# %% [markdown]
# ### Misclassification Errors
# NOTE: NOT WORKING!? ValueError: 
# This solver needs samples of at least 2 classes in the data, 
# but the data contains only one class: 0
# i=0
# for model in models:
#     plot_learning_curves(X_train, y_train, X_test, y_test, model)
#     plt.title('Learning Curve for ' + model_namelist[i], fontsize=14)
#     plt.xlabel('Training Set Size (%)', fontsize=12)
#     plt.ylabel('Misclassification Error', fontsize=12)
#     # plt.show()
#     i += 1

# %% [markdown]
# ### Get predictions: prep for Confusion Matrix
y_test_pred = []
for model in models:
    model.fit(X,y)
    y_test_pred.append(model.predict(X_test))

# %% [markdown]
# ### Graph metrics
fig_size_tuple = (15,7)
title_fontsize_num = 15
label_fontsize_num = 12

df_cross_validate_results = pd.DataFrame(cv_result_entries, columns =['model_name', 'fold_index', 'metric_key', 'metric_score'])

df_cv_results_fit_time = df_cross_validate_results.loc[df_cross_validate_results.metric_key == 'fit_time']
df_cv_results_score_time = df_cross_validate_results.loc[df_cross_validate_results.metric_key == 'score_time']
df_cv_results_accuracy = df_cross_validate_results.loc[df_cross_validate_results.metric_key == 'test_accuracy']
df_cv_results_precision = df_cross_validate_results.loc[df_cross_validate_results.metric_key == 'test_precision']
df_cv_results_recall = df_cross_validate_results.loc[df_cross_validate_results.metric_key == 'test_recall']
df_cv_results_f1 = df_cross_validate_results.loc[df_cross_validate_results.metric_key == 'test_f1']
# df_cv_results_f2 = df_cross_validate_results.loc[df_cross_validate_results.metric_key == 'test_f2']
df_cv_results_roc_auc = df_cross_validate_results.loc[df_cross_validate_results.metric_key == 'test_roc_auc']


plt.figure(figsize=fig_size_tuple)
sns.boxplot(x='model_name', y='metric_score', data = df_cv_results_fit_time)
sns.stripplot(x='model_name', y='metric_score', data = df_cv_results_fit_time, size=10, linewidth=2)
plt.title('Fit Time Model Comparison', fontsize=title_fontsize_num)
plt.xlabel('Model Name', fontsize=label_fontsize_num)
plt.ylabel('Fit Time score', fontsize=label_fontsize_num)
plt.xticks(rotation=45)
# plt.show()
plt.savefig(configs.saveDir +'fit_time.png')

plt.figure(figsize=fig_size_tuple)
sns.boxplot(x='model_name', y='metric_score', data = df_cv_results_score_time)
sns.stripplot(x='model_name', y='metric_score', data = df_cv_results_score_time, size=10, linewidth=2)
plt.title('Score Time Model Comparison', fontsize=title_fontsize_num)
plt.xlabel('Model Name', fontsize=label_fontsize_num)
plt.ylabel('Score Time score', fontsize=label_fontsize_num)
plt.xticks(rotation=45)
# plt.show()
plt.savefig(configs.saveDir + 'rawVoxels_score_time.png')

plt.figure(figsize=fig_size_tuple)
sns.boxplot(x='model_name', y='metric_score', data = df_cv_results_accuracy)
sns.stripplot(x='model_name', y='metric_score', data = df_cv_results_accuracy, size=10, linewidth=2)
plt.title('Accuracy Model Comparison', fontsize=title_fontsize_num)
plt.xlabel('Model Name', fontsize=label_fontsize_num)
plt.ylabel('Accuracy score', fontsize=label_fontsize_num)
plt.xticks(rotation=45)
# plt.show()
plt.savefig(configs.saveDir + 'rawVoxels_accuracy.png')

plt.figure(figsize=fig_size_tuple)
sns.boxplot(x='model_name', y='metric_score', data = df_cv_results_f1)
sns.stripplot(x='model_name', y='metric_score', data = df_cv_results_f1, size=10, linewidth=2)
plt.title('F1 Score Model Comparison', fontsize=title_fontsize_num)
plt.xlabel('Model Name', fontsize=label_fontsize_num)
plt.ylabel('F1 score', fontsize=label_fontsize_num)
plt.xticks(rotation=45)
# plt.show()
plt.savefig(configs.saveDir + 'rawVoxels_f1.png')

# plt.figure(figsize=fig_size_tuple)
# sns.boxplot(x='model_name', y='metric_score', data = df_cv_results_f2)
# sns.stripplot(x='model_name', y='metric_score', data = df_cv_results_f2, size=10, linewidth=2)
# plt.title('F2 Score Model Comparison', fontsize=title_fontsize_num)
# plt.xlabel('Model Name', fontsize=label_fontsize_num)
# plt.ylabel('F2 score', fontsize=label_fontsize_num)
# plt.xticks(rotation=45)
# # plt.show()

plt.figure(figsize=fig_size_tuple)
sns.boxplot(x='model_name', y='metric_score', data = df_cv_results_precision)
sns.stripplot(x='model_name', y='metric_score', data = df_cv_results_precision, size=10, linewidth=2)
plt.title('Precision Model Comparison', fontsize=title_fontsize_num)
plt.xlabel('Model Name', fontsize=label_fontsize_num)
plt.ylabel('Precision score', fontsize=label_fontsize_num)
plt.xticks(rotation=45)
# plt.show()
plt.savefig(configs.saveDir + 'rawVoxels_precision.png')

plt.figure(figsize=fig_size_tuple)
sns.boxplot(x='model_name', y='metric_score', data = df_cv_results_recall)
sns.stripplot(x='model_name', y='metric_score', data = df_cv_results_recall, size=10, linewidth=2)
plt.title('Recall Model Comparison', fontsize=title_fontsize_num)
plt.xlabel('Model Name', fontsize=label_fontsize_num)
plt.ylabel('Recall score', fontsize=label_fontsize_num)
plt.xticks(rotation=45)
# plt.show()
plt.savefig(configs.saveDir + 'rawVoxels_recall.png')

plt.figure(figsize=fig_size_tuple)
sns.boxplot(x='model_name', y='metric_score', data = df_cv_results_roc_auc)
sns.stripplot(x='model_name', y='metric_score', data = df_cv_results_roc_auc, size=10, linewidth=2)
plt.title('ROC-AUC Score Model Comparison', fontsize=title_fontsize_num)
plt.xlabel('Model Name', fontsize=label_fontsize_num)
plt.ylabel('ROC-AUC score', fontsize=label_fontsize_num)
plt.xticks(rotation=45)
# plt.show()
plt.savefig(configs.saveDir + 'rawVoxels_roc-auc.png')

# %% [markdown]
# ### Confusion Matrix

CLASSES = ['Awake', 'Sleepy']
i=0
for _ in models:
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_test_pred[i], axis=1))
    cm_df = pd.DataFrame(cm, index = CLASSES, columns = CLASSES)
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    plt.title('Confusion Matrix for ' + model_namelist[i], fontsize=14)
    sns.heatmap(cm_df, annot=True, fmt='.6g', annot_kws={"size": 10}, cmap='Reds')
    # plt.show()
    plt.savefig(configs.saveDir + 'rawVoxels_confusion_matrix.png')
    i += 1

# %%
