import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import util

from keras.layers import Dense
from keras.models import Sequential    
    
def train_model(X_train, y_train, X_val, y_val, size='small'):
    n_classes = len(util.outcome_classes)
    y_train = keras.utils.to_categorical(y_train, num_classes=n_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes=n_classes)

    if size == 'small':
        h = 16
    elif size == 'medium':
        h = 32
    elif size = 'large':
        h = 64
    else:
        assert(False)
    model = Sequential()
    model.add(Dense(h, activation='relu', input_dim=X_train.shape[1]))
    if size == 'medium':
        model.add(Dense(h, activation='relu'))
    if size = 'large':
        model.add(Dense(h, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    model.fit(x_train, y_train, batch_size=32, epochs=10)
    score = model.evaluate(X_val, y_val)
    
    return model, score


data_split = util.split_indices()
X, y = util.load_data()
models = []
scores = []
for split in data_split:
    xti, yti, xvi, yvi = split
    X_train = X[xti]
    y_train = y[yti]
    X_val = X[xvi]
    y_val = y[yvi]
    model, score = train_model(X_train, y_train, X_val, y_val)
    models.append(model)
    scores.append(score)
    
scores = np.array(scores)
scores = np.mean(scores, axis=0)
print('Validation Results')
print(model.metrics_names)
print(scores)

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
    
    




