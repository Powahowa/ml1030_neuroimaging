# %%
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest

import configurations

configs = configurations.Config('sub-xxx-resamp-intersected')

# %%
X = np.load('../features/rawFunctionalConnectivity-STCM.npy', allow_pickle=True)
y = np.load('../features/labels.npy', allow_pickle=True)

# %%
print(X)
print(X.shape)
print(X[0])
print(X[0].shape)
print(X[0][0])
print(X[0][0].shape)
print(X[0][0][0])

# %%
def slice_array(arr, start, end):
    sliced_arr = arr[start:end]
    return sliced_arr

shapes_list = [X[i].shape for i in range(len(X))]
min_shape = min(shapes_list)

if (configs.endSlice - configs.startSlice) < min_shape[0]:
    sliced_X_list = []
    for item in X:
        sliced_X_list.append(slice_array(item, configs.startSlice, configs.endSlice))
    X = np.stack(sliced_X_list)
else:
    err = str('Chosen slice size ' +
              str(configs.endSlice - configs.startSlice) + 
              ' is larger size of smallest array ' +
              str(min_shape[0])
    )
    raise Exception(err)

# %%
# ## Define LSTM
lstm_size = 10
def get_lstm():
    # input_shape = (configs.endSlice-configs.startSlice, len(X[0,0,:]))
    num_classes = 1
    keras.backend.clear_session()
    
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(lstm_size, activation='relu'))
    # model.add(keras.layers.Dropout(0.25))
    # model.add(keras.layers.LSTM(2, 'relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(1e-4), 
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
        monitor='val_loss',
        verbose=verbosity, 
        save_weights_only=False, 
        save_best_only=True
    )
    early = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        mode='min', 
        patience=num_waits
    )
    tb = keras.callbacks.TensorBoard(log_dir=logdir)
    # callbacks_list = [checkpoint, early, tb]
    callbacks_list = [checkpoint, early]

    model = get_lstm()
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
    trainloss, trainacc = model.evaluate(x_train, y_train, verbose=1)
    valloss, valacc = model.evaluate(x_val, y_val, verbose=1)
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
# %%
