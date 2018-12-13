import util
import numpy as np 
from collections import Counter
from sklearn.model_selection import train_test_split


outcome_column_num = 2
hostlev_column_num = 5
y_label_columns = [outcome_column_num, hostlev_column_num]


data = np.genfromtxt("../data/disputesBordersTimeSorted.csv", dtype=float, delimiter=',', skip_header=1)[:, 1:]
data = util.insert_alliance_features(data)

# remap labels --> victory = 1, yield = 2, compromise = 3, stalemate = 4, other = 5
for i, row in enumerate(data):
	# victory	
	if row[outcome_column_num] == 1 or row[outcome_column_num] == 2:
		data[i][outcome_column_num] = 1
	# yield
	elif row[outcome_column_num] == 3 or row[outcome_column_num] == 4:
		data[i][outcome_column_num] = 2
	# compromise
	elif row[outcome_column_num] == 6:
		data[i][outcome_column_num] = 3
	# stalemate
	elif row[outcome_column_num] == 5:
		data[i][outcome_column_num] = 4	
	# other
	else:
		data[i][outcome_column_num] = 5

y = data[:, y_label_columns]
X = np.delete(data, y_label_columns, axis=1)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
num_examples = X.shape[0]
print(X.shape)
indices = list(range(num_examples))
train_indices = indices[ : int(num_examples * .8)]
test_indices = indices[int(num_examples * .8) : ]

X_train_time = X[train_indices]
y_train_time = y[train_indices]

X_test_time = X[test_indices]
y_test_time = y[test_indices]

print(X_train_time.shape)
print(X_test_time.shape)

util.dumpVar("../data/X_train_time", X_train_time)
util.dumpVar("../data/y_train_time", y_train_time)

util.dumpVar("../data/X_test_time", X_test_time)
util.dumpVar("../data/y_test_time", y_test_time)