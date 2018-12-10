import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import util
from keras import __version__ as used_keras_version
print(used_keras_version)
import matplotlib.pyplot as plt

from keras import regularizers
from keras.layers import Dense
from keras.models import Sequential 

def normalize(a):
    if a is None:
        return None
    return (a - np.mean(a, axis=0)) / (np.std(a, axis=0) + 1e-5)
    
def plot_history(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
def train_model(X_train, y_train, X_val, y_val, layers=2, plot=False, reg_weight=0.0):
    X_train = normalize(X_train)
    X_val = normalize(X_val)
    n_classes = len(util.outcome_classes)
    y_train = keras.utils.to_categorical(y_train, num_classes=n_classes)
    if y_val is not None:
        y_val = keras.utils.to_categorical(y_val, num_classes=n_classes)

    h = max(256, 8 * 2 ** int(layers / 2))
    model = Sequential()
    model.add(Dense(h, activation='relu', 
        kernel_regularizer=regularizers.l2(reg_weight),
        input_dim=X_train.shape[1]))
    for i in range(layers - 1):
        model.add(Dense(h, 
            kernel_regularizer=regularizers.l2(reg_weight), 
            activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
    
    if X_val is None or y_val is None:
        history = model.fit(X_train, y_train, validation_split=0.1, batch_size=32, epochs=10)
    else:
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=1)
    
    if X_val is not None and y_val is not None:
        score = model.evaluate(X_val, y_val)
        return model, score
    else:
        if plot: 
            plot_history(history)
        return model, history


X_train = util.openPkl("../data/X_train")
y_train = util.openPkl("../data/y_train")[:, 0].astype('int') - 1
model = train_model(X_train, y_train, None, None, layers=4, plot=True)

accs = []
reg_weights = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.]
for w in reg_weights:
    _, history = train_model(X_train, y_train, None, None, layers=4, plot=False, reg_weight=w)
    accs.append(max(history.history['val_categorical_accuracy']))

plt.plot(reg_weights, accs)
plt.ylabel('accuracy')
plt.xlabel('regularization weight')
plt.xscale('log')
plt.show()
print('')

accs = []
layers_list = range(2, 10 + 1)
for layers in layers_list:
    _, history = train_model(X_train, y_train, None, None, layers=layers, plot=False)
    accs.append(max(history.history['val_categorical_accuracy']))

plt.plot(list(layers_list), accs)
plt.ylabel('accuracy')
plt.xlabel('model size (layers)')
plt.show()
print('')

'''
data_split = util.split_data()
models = []
scores = []
all_X_val = None
all_y_val = None
for split in data_split:
    X_train, y_train, X_val, y_val = split
    y_train = y_train.astype('int') - 1
    y_val = y_val.astype('int') - 1
    model, score = train_model(X_train, y_train, X_val, y_val)
    print('')
    models.append(model)
    scores.append(score)
    print(model.predict(X_train))
    
    if all_X_val is None:
        all_X_val = X_val
        all_y_val = y_val
    else:
        all_X_val = np.vstack((all_X_val, X_val))
        all_y_val = np.concatenate((all_y_val, y_val))

print("Val Results")
predictions = []
for model in models:
    y_hat = model.predict(X_val)
    predictions.append(y_hat)
    
predictions = np.array(predictions)
y_hat = np.mean(predictions, axis=0)
y_hat = np.argmax(y_hat, axis=1)
print(y_hat)
print('Accuracy = {}'.format(np.mean(y_hat == y_val)))
'''

'''
print("Test Results")
predictions = []
for model in models:
    y_hat = model.predict(x_test)
    predictions.append(y_hat)

predictions = np.array(predictions) #3d array
y_hat = np.mean(predictions, axis=0)
y_hat = np.argmax(y_hat, axis=1)
print('Accuracy = {}'.format(np.mean(y_hat == y_test)))
    
'''
    
    




