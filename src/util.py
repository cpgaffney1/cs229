import pandas as pd
import numpy as np
import pickle
import itertools
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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

classes = np.asarray(list(range(1,6)))

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
