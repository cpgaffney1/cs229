import util
from sklearn.model_selection import train_test_split
import numpy as np 
from collections import Counter


# FILL THIS IN/UPDATE ONCE DATASET IS FINALIZED
outcome_column_num = 2
hostlev_column_num = 4
y_label_columns = [outcome_column_num, hostlev_column_num]


# FILL IN CSV FILENAME LATER
data = np.genfromtxt("../data/data_not_final.csv", dtype=float, delimiter=',', skip_header=1)[:, 1:]

# remap labels --> victory = 1, yield = 2, compromise = 3, stalemate = 4, other = 5

# keep index dict to take random subset of larger classes
class_four = []

for i, row in data:
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
		class_four.append(i)
	# other
	else:
		data[i][outcome_column_num] = 5

y = data[:, y_label_columns]
X = np.delete(X, y_label_columns, axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


util.dumpVar("../data/X_train", X_train)
util.dumpVar("../data/y_train", y_train)

util.dumpVar("../data/X_test", X_test)
util.dumpVar("../data/y_test", y_test)