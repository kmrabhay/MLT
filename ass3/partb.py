import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

Xtest = pd.read_csv("Xtest.csv", header=None).as_matrix()
Xtrain = pd.read_csv("Xtrain.csv", header=None).as_matrix()
Ytest = pd.read_csv("label_test.csv", header=None).as_matrix().flatten()
Ytrain = pd.read_csv("label_train.csv", header=None).as_matrix().flatten()

hog_feature_train = []
for t in Xtrain:
	fh = hog(t.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
	hog_feature_train.append(fh)

hog_features_train_array = np.array(hog_feature_train, 'float64')

for ntree in range(5,50):

	clf = RandomForestClassifier(n_estimators=ntree)  ### part(b) ####
	clf.fit(hog_features_train_array, Ytrain)
	hog_feature_test = []

	for t in Xtest:
		fh = hog(t.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
		hog_feature_test.append(fh)

	hog_feature_test_array =np.array(hog_feature_test, 'float64')
	nbr = clf.predict(hog_feature_test_array)
	count = 0

	for i in range(0, len(Ytest)):
		if nbr[i]== Ytest[i]:
			count+=1
	print ' no of tree ', ntree, ' no. of matched', count, ' acuracy is', count/10,'%' 