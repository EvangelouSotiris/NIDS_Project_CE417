import keras
import sys
from dataset_prepare import prepare_data
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import keras.backend as K

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

	model.add(Dense(20, input_dim=trainx.shape[1], activation='relu'))
	
	#model.add(Dense(10, activation='relu'))

	model.add(Dense(1, activation='hard_sigmoid'))

	our_rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay = 0.0)

	model.compile(optimizer=our_rmsprop,loss='binary_crossentropy',metrics=['mse','accuracy'])
	return model

scaler = MinMaxScaler(feature_range=(0, 1))
trainx,trainy,testx,testy,features = prepare_data()

trainx = scaler.fit_transform(trainx)
testx = scaler.fit_transform(testx)

model = create_model(trainx)

hist = model.fit(trainx, trainy, epochs= 300, batch_size=1000 , shuffle=True)

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