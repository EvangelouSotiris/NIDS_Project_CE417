import keras
import sys

from dataset_prepare import prepare_data

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt


def random_forest():
	"""
	This function performs classification with random forest.
	In order to call it please run with "$ python3 ids.py randomforest"

	"""
	train_x,train_y,test_x,test_y ,axristo= prepare_data()
	train_y = train_y.reshape((train_y.shape[0],))

	clf = RandomForestClassifier(n_estimators=1000, max_depth=5,random_state=None)
	clf.fit(train_x,train_y)

	print("Accuracy: {}%".format(clf.score(test_x,test_y)*100))


def supported_vector_machine():
	"""
	This function performs classification with svm.
	In order to call it please run with "$ python3 ids.py svm"
	So far, we have achieved a high point 73% accuracy with this method.

	"""
	train_x,train_y,test_x,test_y,column_names = prepare_data()

	C=10
	gamma =0.1

	train_y = train_y.reshape((train_y.shape[0],))

	print(C,gamma)
	classifier = SVC(kernel="rbf",C=C,gamma=gamma, verbose=1)
	classifier.fit(train_x[:30000,:], train_y[:30000])
	SVs = classifier.support_vectors_ #support vectors

	print("For C : ",C," Gamma: ",gamma) 
	print("Number of Support Vectors: %d" %len(SVs))
	print('\n')
	print("Accuracy: {}%".format(classifier.score(test_x[:30000,:], test_y[:30000]) * 100 ))


def graph_accuracy(acc):
	plt.subplot(1, 2, 1)
	plt.title("Accuracy in every iteration of training")
	plt.plot(acc, 'r-', label="accuracy")
	plt.legend(loc='lower right')
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
	
	model.add(Dense(50, activation='sigmoid'))

	model.add(Dense(1, activation='sigmoid'))

	our_rmsprop = keras.optimizers.RMSprop(lr=0.05, rho=0.9, epsilon=None, decay = 0.0001)

	model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['mse','accuracy'])
	return model

def multilayer_perceptron():
	scaler = MinMaxScaler(feature_range=(0, 1))
	trainx,trainy,testx,testy,features = prepare_data()

	trainx = scaler.fit_transform(trainx)
	testx = scaler.fit_transform(testx)

	model = create_model(trainx)

	es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)

	hist = model.fit(trainx, trainy, epochs= 150, batch_size=10 , shuffle=True, callbacks=[es])

	plt.figure(figsize=(50,50))
	graph_accuracy(hist.history['acc'])
	graph_loss(hist.history['mean_squared_error'])
	plt.show()

	ypreds = model.predict(testx)
	for i in range(len(ypreds)):
		ypreds[i] = hard_lim(ypreds[i])

	confusion_matrix_metric = confusion_matrix(testy, ypreds)
	print("\nConfusion Matrix for test dataset.")
	print(confusion_matrix_metric)

	score = model.evaluate(testx, testy, batch_size=50)
	print("\nBinary crossentropy :%s \nMean Squared Error:%s \nAccuracy:%s %%" %(score[0],score[1],score[2]))

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