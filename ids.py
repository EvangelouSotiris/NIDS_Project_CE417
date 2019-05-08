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

	print("\n## Time ##")
	print("Training lasted {} seconds".format(end-start))
	print("\n## Accuracies ##")
	print("Normal/Abnormal activity accuracy:{}%".format((correct_class/len(labels))*100))
	print("Attack type classification accuracy: {}%".format(clf.score(test_x,test_y)*100))

	return(clf.score(test_x,test_y)*100,(correct_class/len(labels))*100)

def supported_vector_machine():
	"""
	This function performs classification with svm.
	In order to call it please run with "$ python3 ids.py svm"
	So far, we have achieved a high point 73% accuracy with this method.

	"""
	train_x,train_y,test_x,test_y,labels = prepare_data()

	C=100
	gamma =0.00001
	boundary = 20000

	train_y = train_y.reshape((train_y.shape[0],))

	print(C,gamma)
	classifier = SVC(kernel="rbf",C=C,gamma=gamma,verbose=0)

	start = time.time()
	classifier.fit(train_x[:boundary,:], train_y[:boundary])
	end = time.time()


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

	SVs = classifier.support_vectors_ #support vectors
	print("For C : ",C," Gamma: ",gamma) 
	print("Number of Support Vectors: %d" %len(SVs))
	print('\n')

	print("\n## Time ##")
	print("Training lasted {} seconds".format(end-start))
	print("\n## Accuracies ##")
	print("Normal/Abnormal activity accuracy:{}%".format((correct_class/len(yhat))*100))
	print("Attack type classification accuracy: {}%".format(classifier.score(test_x, test_y) * 100 ))

	return(classifier.score(test_x, test_y) * 100,(correct_class/len(yhat))*100)

def graph_accuracy(acc,test_acc):
	plt.subplot(1, 2, 1)
	plt.title("Accuracy in every iteration of training")
	plt.plot(acc, 'r-', label="training accuracy")
	plt.plot(len(acc),test_acc,'b*', label="testing accuracy",markersize=12)
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

def graph_confusion_matrix(conf):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(conf)
	plt.title('Confusion matrix of Normal/Abnormal traffic classification')
	fig.colorbar(cax)
	ax.set_xticklabels([''] + ['True','False'])
	ax.set_yticklabels([''] + ['True','False'])
	ax.text(0, 0, "True-Positive",ha="center", va="center", color="green")
	ax.text(0, 1, "False-Positive",ha="center", va="center", color="coral")
	ax.text(1, 0, "False-Negative",ha="center", va="center", color="coral")
	ax.text(1, 1, "True-Negative",ha="center", va="center", color="green")
	plt.show()

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
	plt.style.use('seaborn-muted')
	plt.style.context(('dark_background'))

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
	hist = model.fit(trainx, trainy, epochs= 150, batch_size=150 , shuffle=True,verbose = 1)#, callbacks=[es])
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
	print("\n## Time ##")
	print("Training lasted {} seconds".format(end-start))

	print("\n## Accuracies ##")
	print("Normal/Abnormal activity accuracy:{}%".format((correct_class/len(labels))*100))

	score = model.evaluate(testx, testy, batch_size=50)
	print("Attack type classification:\nCategorical crossentropy :%s \nMean Squared Error:%s \nAccuracy:%s %%" %(score[0],score[1],score[2]*100))

	confusion_matrix_metric = confusion_matrix(labels, predictions)
	print("\n## Confusion Matrix ##")
	print("Confusion Matrix for test dataset.")
	print(confusion_matrix_metric)

	plt.figure(figsize=(50,50))
	graph_accuracy(hist.history['acc'],score[2])
	graph_loss(hist.history['mean_squared_error'])
	plt.show()

	graph_confusion_matrix(confusion_matrix_metric)

	return(score[2]*100,(correct_class/len(labels))*100)

# MAIN
if __name__ == '__main__':
	# Define the user's preferred method
	if len(sys.argv) > 1:
		if sys.argv[1] == 'svm':
			supported_vector_machine()
		elif sys.argv[1] == 'randomforest':
			random_forest()
		elif sys.argv[1] == '--comparative':
			svm_atk,svm_lab = supported_vector_machine()
			rf_atk, rf_lab = random_forest()
			mlp_atk, mlp_lab = multilayer_perceptron()
			labels = [svm_lab, rf_lab, mlp_lab]
			atk_cats = [svm_atk, rf_atk, mlp_atk]
			plt.ylim(0, 100)
			l1,l2,l3 = plt.bar(["SVM-norm", "RF-norm", "MLP-norm"],labels)
			c1,c2,c3 = plt.bar(["SVM-atk" , "RF-atk", "MLP-atk"],atk_cats)
			l1.set_facecolor('r')
			c1.set_facecolor('b')
			l2.set_facecolor('r')
			c2.set_facecolor('b')
			l3.set_facecolor('r')
			c3.set_facecolor('b')
			plt.show()
	else:
		multilayer_perceptron()