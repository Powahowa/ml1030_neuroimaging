# %%
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest

from joblib import Parallel, delayed

import configurations

configs = configurations.Config('STCM_confoundsOut_43-103slice')

import tensorflow as tf

print("TF GPU list:", tf.config.experimental.list_physical_devices('GPU'))

print ("TF GPU Build test:", tf.test.is_built_with_cuda())

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
    num_filters = [24,32,64,128] 
    pool_size = (2) 
    kernel_size = (3)  
    input_shape = (configs.endSlice-configs.startSlice, len(X[0,0,:]))
    num_classes = 1
    keras.backend.clear_session()
    
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(24, kernel_size, input_shape=input_shape,
                padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling1D(pool_size=pool_size))

    model.add(keras.layers.Conv1D(32, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))  
    model.add(keras.layers.MaxPooling1D(pool_size=pool_size))
    
    model.add(keras.layers.Conv1D(64, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))  
    model.add(keras.layers.MaxPooling1D(pool_size=pool_size))
    
    model.add(keras.layers.Conv1D(128, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))  

    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(num_classes, activation="sigmoid"))

    model.compile(optimizer=keras.optimizers.Adam(1e-3), 
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
logdir = './CNNv2/'
num_epochs = 100
num_waits = 10
verbosity = 1

for train_index, test_index in kf.split(folds):
    # x_train = np.array(X.iloc[folds[train_index]])
    # y_train = np.array(y.iloc[folds[train_index]])

    # x_val = np.array(X.iloc[folds[test_index]])
    # y_val = np.array(y.iloc[folds[test_index]])
    x_train = X[folds[train_index]]
    y_train = y[folds[train_index]]

    x_val = X[folds[test_index]]
    y_val = y[folds[test_index]]

    # Checkpoint to continue models, early stopping and tensorboard
    checkpoint = keras.callbacks.ModelCheckpoint(
        logdir + 'best_%d.h5'%i, 
        monitor='val_loss',
        verbose=verbosity, 
        save_weights_only=True, 
        save_best_only=True
    )
    early = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        mode='min', 
        patience=num_waits
    )
    tb = keras.callbacks.TensorBoard(log_dir=logdir)
    callbacks_list = [checkpoint, early, tb]
    # callbacks_list = [checkpoint, early]

    model = get_cnn()
    # model.fit(x_train, y_train, epochs = 10, batch_size = 24, verbose = 0)
    history_cnn = model.fit(
        x_train, 
        y_train, 
        epochs=num_epochs,
        use_multiprocessing=False, 
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

pd.DataFrame.from_dict(history_cnn.history).to_csv('./CNNv2/rawVoxelsHistory.csv',index=False)
model.save('./CNNv2/rawVoxelModel')
# %%
