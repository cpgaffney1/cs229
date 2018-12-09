import pandas as pd
import numpy as np
import pickle
import itertools
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.model_selection import KFold

classes = np.asarray(list(range(1,6)))
K = 10

# indices into y_train and y_test
outcome_col_index = 0
hostlev_col_index = 1

def get_country_mapping():
    # returns mapping of country code to name
    table = pd.read_csv('data/country_codes.csv')
    codes_to_names = {}
    for i in range(len(table)):
        name = table['StateNme'].iloc[i]
        code = table['CCode'].iloc[i]
        codes_to_names[code] = name
    return codes_to_names

def dumpVar(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def openPkl(filename):
    var = open(filename, "rb")
    return pickle.load(var)

outcome_classes = np.asarray(list(range(1,6)))

def outputConfusionMatrix(pred, labels, filename):
    """ Generate a confusion matrix """
    
    cm = confusion_matrix(labels, pred, labels=classes)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)  

    
openPkl('data/alliance_features/features1816.pickle')

def load_alliance_graph(year):
    assert(year >= 1816)
    assert(year <= 2012)
    G = snap.TUNGraph.Load(snap.TFIn('data/alliances/alliances{}.graph'.format(year)))
    return G
    

    

# Selects random sample of original data s.t. classes are balanced
# Splits into K folds
# Defaults to using column index for outcome
# Returns list of length K where each tuple is one fold: (X_train, y_train, X_val, y_val)
def split_data(y_index=outcome_col_index):
    X_train = openPkl("../../X_train_temp")
    y_train = openPkl("../../y_train_temp")[:, y_index]
        
    # balance dataset
    class_counts = {c : [] for c in classes }
    for index, label in enumerate(y_train):
        class_counts[label].append(index)
    assert(sum([len(class_counts[c]) for c in class_counts]) == y_train.shape[0])
    # print(sorted(Counter(list(y_train)).items()))

    max_num_values = min([len(class_counts[c]) for c in class_counts])
    # print("max_num_values:", max_num_values)
    
    indices_to_delete = []
    delete_me = {c : 0 for c in classes }
    for label in class_counts:
        label_indices = class_counts[label]
        num_to_delete = len(label_indices) - max_num_values
        delete_me[label] = num_to_delete
        if num_to_delete > 0:
            indices_to_delete += list(np.random.choice(label_indices, size=num_to_delete, replace=False))
    assert(len(indices_to_delete) == sum([delete_me[label] for label in delete_me]))
    # print("elements to delete:", len(indices_to_delete))
    new_X = np.delete(X_train, indices_to_delete, axis=0)
    new_y = np.delete(y_train, indices_to_delete, axis=0)
    # print("before:", X_train.shape)
    # print("after:", new_X.shape)
    # print(sorted(Counter(list(new_y)).items()))
    
    # new_indices = {c : [] for c in classes}
    # for index, label in enumerate(new_y):
    #     new_indices[label].append(index)
    # print(new_indices)   

    # split into K blocks
    kf = KFold(n_splits=K)
    splits = []
    for train_indices, val_indices in kf.split(new_X):
        splits.append((new_X[train_indices], new_y[train_indices], new_X[val_indices], new_y[val_indices]))
        # print((train_indices))
        # print((val_indices))
    return splits



