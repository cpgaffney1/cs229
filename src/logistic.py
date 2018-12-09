import util
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


y_label_index = util.outcome_col_index

X_train = util.openPkl("../../X_train_temp")
y_train = util.openPkl("../../y_train_temp")[:, y_label_index] 

n_features = X_train.shape[1]
n_classes = len(util.classes)

avg_coefficients = np.zeros((n_classes, n_features))
avg_intercepts = np.zeros(n_classes)

data_kfold = util.split_data(y_index=y_label_index)
train_accuracies = []
eval_accuracies = []
train_predictions = []
eval_predictions = []

for i, (X_train, y_train, X_val, y_val) in enumerate(data_kfold):
	print("Fold", i+1)
	clf = LogisticRegression(max_iter=5000, multi_class= 'multinomial', solver= 'newton-cg')
	clf.fit(X_train, y_train)
	train_acc = clf.score(X_train, y_train)
	train_accuracies.append(train_acc)
	eval_acc = clf.score(X_val, y_val)
	eval_accuracies.append(eval_acc)
	avg_coefficients += clf.coef_
	avg_intercepts += clf.intercept_
	train_predictions.append(clf.predict(X_train))
	eval_predictions.append(clf.predict(X_val))

	print("train accuracy:", train_acc)
	print("eval accuracy:", eval_acc)

avg_coefficients /= util.K
avg_intercepts /= util.K 
model = {
	"coeff_": avg_coefficients,
	"intercept_": avg_intercepts,
	"train_accuracies": train_accuracies,
	"eval_accuracies": eval_accuracies,
	"train_predictions": train_predictions,
	"eval_predictions": eval_predictions
}

util.dumpVar("avg_logistic_model", model)

# model = util.openPkl("logistic_model")
# avg_coefficients = model['coeff_']
# avg_intercepts = model['intercept_']

clf = LogisticRegression()
clf.coef_ = avg_coefficients
clf.intercept_ = avg_intercepts
clf.classes_ = util.classes
print("Averaged model train accuracy:", clf.score(X_train, y_train))
