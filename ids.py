import sys
import time

from dataset_prepare import prepare_data

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.utils import np_utils as npu
from keras import optimizers

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt

import numpy as np


def random_forest():
	"""
	This function performs classification with random forest.
	In order to call it please run with "$ python3 ids.py randomforest"

	"""
	train_x,train_y,test_x,test_y ,labels= prepare_data()
	train_y = train_y.reshape((train_y.shape[0],))

	clf = RandomForestClassifier(n_estimators=1000, max_depth=5,random_state=None)
	
	start = time.time()
	clf.fit(train_x,train_y)
	end = time.time()

	yhat = clf.predict(test_x)
	for i in range(len(yhat)):
		if yhat[i] == 6:
			yhat[i] = 0 
		else:
			yhat[i] = 1
	correct_class = 0
	for i in range(len(labels)):
		if labels[i] == yhat[i]:
			correct_class += 1
	print("Training lasted {} seconds".format(end-start))
	print("Normal/Abnormal activity accuracy:{}%".format((correct_class/len(labels))*100))
	print("Attack type accuracy: {}%".format(clf.score(test_x,test_y)*100))


def supported_vector_machine():
	"""
	This function performs classification with svm.
	In order to call it please run with "$ python3 ids.py svm"
	So far, we have achieved a high point 73% accuracy with this method.

	"""
	train_x,train_y,test_x,test_y,labels = prepare_data()

	C=100
	gamma =0.00001
	boundary = 50000

	train_y = train_y.reshape((train_y.shape[0],))

	print(C,gamma)
	classifier = SVC(kernel="rbf",C=C,gamma=gamma,verbose=0)
	classifier.fit(train_x[:boundary,:], train_y[:boundary])

	yhat = classifier.predict(test_x)

	for i in range(len(yhat)):
		if yhat[i] == 6:
			yhat[i] = 0 
		else:
			yhat[i] = 1
	correct_class = 0

	for i in range(len(yhat)):
		if labels[i] == yhat[i]:
			correct_class += 1
	print("accuracy:{}%".format((correct_class/len(yhat))*100))

	SVs = classifier.support_vectors_ #support vectors

	print("For C : ",C," Gamma: ",gamma) 
	print("Number of Support Vectors: %d" %len(SVs))
	print('\n')
	#print("Accuracy: {}%".format(classifier.score(test_x, test_y) * 100 ))


def graph_accuracy(acc,test_acc):
	plt.subplot(1, 2, 1)
	plt.title("Accuracy in every iteration of training")
	plt.plot(acc, 'r-', label="training accuracy")
	plt.bar(len(acc),test_acc, label="testing accuracy")
	plt.legend(loc='lower left')
	plt.xlabel("Iterations")
	plt.ylabel("Accuracy")

def graph_loss(loss):
	plt.subplot(1, 2, 2)
	plt.title("Mean Squared Error in every iteration of training")
	plt.plot(loss, 'b-', label="loss")
	plt.legend(loc='upper right')
	plt.xlabel("Iterations")
	plt.ylabel("Mean Squared Error")

def hard_lim(x):
	if x < 0.5:
		return 0
	else:
		return 1

def create_model(trainx):
	model = Sequential()

	model.add(Dense(100, input_dim=trainx.shape[1], activation='relu'))
	
	model.add(Dense(50, activation='relu'))

	model.add(Dense(10, activation='softmax'))

	model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['mse','accuracy'])
	return model

def multilayer_perceptron():
	scaler = MinMaxScaler(feature_range=(0, 1))
	trainx,trainy,testx,testy,labels = prepare_data()
	temp = testy
	trainx = scaler.fit_transform(trainx)
	testx = scaler.fit_transform(testx)

	trainy = npu.to_categorical(trainy)
	testy = npu.to_categorical(testy)

	es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)

	model = create_model(trainx)

	start = time.time()
	hist = model.fit(trainx, trainy, epochs= 5, batch_size=150 , shuffle=True,verbose = 1)#, callbacks=[es])
	end = time.time()

	ypreds = model.predict(testx)
	predictions = np.argmax(ypreds, axis=1)

	for i in range(len(predictions)):
		if predictions[i] == 6:
			predictions[i] = 0
		else:
			predictions[i] = 1

	correct_class = 0
	for i in range(len(labels)):
		if labels[i] == predictions[i]:
			correct_class += 1

	print("Training lasted {} seconds".format(end-start))
	print("Normal/Abnormal activity accuracy:{}%".format((correct_class/len(labels))*100))
	
	confusion_matrix_metric = confusion_matrix(labels, predictions)
	print("\nConfusion Matrix for test dataset.")
	print(confusion_matrix_metric)

	score = model.evaluate(testx, testy, batch_size=50)
	print("Attack type classification:\nCategorical crossentropy :%s \nMean Squared Error:%s \nAccuracy:%s %%" %(score[0],score[1],score[2]))


	plt.figure(figsize=(50,50))
	graph_accuracy(hist.history['acc'],score[2])
	graph_loss(hist.history['mean_squared_error'])
	plt.show()


	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(confusion_matrix_metric)
	plt.title('Confusion matrix of the classifier')
	fig.colorbar(cax)
	ax.set_xticklabels([''] + ['True','False'])
	ax.set_yticklabels([''] + ['True','False'])
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.show()

# MAIN
if __name__ == '__main__':
	# Define the user's preferred method
	if len(sys.argv) > 1:
		if sys.argv[1] == 'svm':
			supported_vector_machine()
		elif sys.argv[1] == 'randomforest':
			random_forest()
	else:
		multilayer_perceptron()