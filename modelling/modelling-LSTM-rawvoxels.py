# %%
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest

from joblib import Parallel, delayed

import configurations

configs = configurations.Config('sub-xxx-resamp-intersected')

# %%
raw_df = pd.read_pickle(configs.rawVoxelFile)
#the below hard coded value is the 2nd dimension of the raw feature file
df2 = raw_df
y = np.asarray(raw_df['sleepdep'])
df2 = df2.drop(columns=['sleepdep'])
tmp_list = []
for row_index in range(len(df2)):
    tmp_list.append(np.reshape(np.array(df2.iloc[row_index]), newshape=(configs.endSlice-configs.startSlice, -1)))
X = np.stack(tmp_list, axis=0)

# %%
# ## Define CNN
def get_cnn():
    input_shape = (configs.endSlice-configs.startSlice, len(X[0,0,:]))
    num_classes = 1
    keras.backend.clear_session()
    
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(100, activation='relu', input_shape=input_shape))
    model.add(keras.layers.Dense(num_classes, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(), 
        loss=keras.losses.BinaryCrossentropy(), 
        metrics=["accuracy"])
    return model

# %% [markdown]
# ### Train and evaluate via 10-Folds cross-validation
accuracies = []
folds = np.array(list(range(len(X))))
kf = KFold(n_splits=5)
trainaccuracies = []
valaccuracies = []
testaccuracies = []
i = 0
logdir = './LSTM/'
num_epochs = 1000
num_waits = 100
verbosity = 1

for train_index, test_index in kf.split(folds):
    x_train = X[folds[train_index]]
    y_train = y[folds[train_index]]

    x_val = X[folds[test_index]]
    y_val = y[folds[test_index]]

    # Checkpoint to continue models, early stopping and tensorboard
    checkpoint = keras.callbacks.ModelCheckpoint(
        logdir + 'best_%d.h5'%i, 
        monitor='accuracy',
        verbose=verbosity, 
        save_weights_only=True, 
        save_best_only=True
    )
    early = keras.callbacks.EarlyStopping(
        monitor='accuracy', 
        mode='max', 
        patience=num_waits
    )
    tb = keras.callbacks.TensorBoard(log_dir=logdir)
    # callbacks_list = [checkpoint, early, tb]
    callbacks_list = [checkpoint, early]

    model = get_cnn()
    # model.fit(x_train, y_train, epochs = 10, batch_size = 24, verbose = 0)
    history_cnn = model.fit(
        x_train, 
        y_train, 
        epochs=num_epochs,
        use_multiprocessing=True, 
        verbose=0,
        callbacks=callbacks_list,
        validation_data=(x_val, y_val)
    )
    trainloss, trainacc = model.evaluate(x_train, y_train, verbose=0)
    valloss, valacc = model.evaluate(x_val, y_val, verbose=0)
    # testloss, testacc = model.evaluate(x_test, y_test, verbose=0)
    trainaccuracies.append(trainacc)
    valaccuracies.append(valacc)
    # testaccuracies.append(testacc)
    print(f"Fold: {i}")
    print("Train Loss: {0} | Accuracy: {1}".format(trainloss, trainacc))
    print("Val Loss: {0} | Accuracy: {1}".format(valloss, valacc))
    # print("Test Loss: {0} | Accuracy: {1}".format(testloss, testacc))
    i += 1

# Out of loop, print average of the results
print("===============================================")
print("FINISHED!")
print(f"Number of Epochs per fold: {num_epochs}")
print("Average Train 10 Folds Accuracy: {0}".format(np.mean(trainaccuracies)))
print("Average Val 10 Folds Accuracy: {0}".format(np.mean(valaccuracies)))
# print("Average Test 10 Folds Accuracy: {0}".format(np.mean(testaccuracies)))

pd.DataFrame.from_dict(history_cnn.history).to_csv('./LSTM/rawVoxelsHistory.csv',index=False)
model.save('./LSTM/rawVoxelModel')
# %%
