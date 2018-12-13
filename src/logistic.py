import util
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def trainAndSaveModel(X_train, y_train, y_label_index, max_iterations=7000, folds=False):
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
		clf = LogisticRegression(max_iter=max_iterations, multi_class= 'multinomial', solver= 'newton-cg')
		clf.fit(X_train, y_train)
		train_acc = clf.score(X_train, y_train)
		train_accuracies.append(train_acc)
		eval_acc = clf.score(X_val, y_val)
		eval_accuracies.append(eval_acc)
		avg_coefficients += clf.coef_
		avg_intercepts += clf.intercept_
		train_predictions.append(clf.predict(X_train))
		eval_predictions.append(clf.predict(X_val))
		if folds:
			util.outputConfusionMatrix(clf.predict(X_train), y_train, "../figures/fold_" + str(i+1) + "_train")
			util.outputConfusionMatrix(clf.predict(X_val), y_val, "../figures/fold_" + str(i+1) + "_eval")

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

	util.dumpVar("../models/avg_logistic_model", model)



def examineModel(X_train, y_train):
	model = util.openPkl("../models/avg_logistic_model")
	avg_coefficients = model['coeff_']
	avg_intercepts = model['intercept_']

	clf = LogisticRegression()
	clf.coef_ = avg_coefficients
	clf.intercept_ = avg_intercepts
	clf.classes_ = util.classes
	print("Averaged model train accuracy:", clf.score(X_train, y_train))
	avg_preds = clf.predict(X_train)
	# util.outputConfusionMatrix(avg_preds, y_train, "../figures/avg_logistic.png")			



if __name__ == '__main__':	
	y_label_index = util.outcome_col_index

	X_train = util.openPkl("../data/X_train")
	y_train = util.openPkl("../data/y_train")[:, y_label_index] 
	
	trainAndSaveModel(X_train, y_train, y_label_index, max_iterations=7000, folds=False)
	examineModel(X_train, y_train)
	model = util.openPkl("../models/avg_logistic_model")
	print(np.average(model["train_accuracies"]))
	print(np.average(model["eval_accuracies"]))
	
	# avg_coefficients = model["coeff_"]
	# n_classes x n_features
	# top_features = np.argmax(avg_coefficients, axis=1)
	# print(top_features)


