import util
import numpy as np 
from collections import Counter
from sklearn.model_selection import train_test_split


outcome_column_num = 2
hostlev_column_num = 5
y_label_columns = [outcome_column_num, hostlev_column_num]


data = np.genfromtxt("../data/final_data.csv", dtype=float, delimiter=',', skip_header=1)[:, 1:]
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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

print(X_train.shape)

util.dumpVar("../data/X_train", X_train)
util.dumpVar("../data/y_train", y_train)

util.dumpVar("../data/X_test", X_test)
util.dumpVar("../data/y_test", y_test)