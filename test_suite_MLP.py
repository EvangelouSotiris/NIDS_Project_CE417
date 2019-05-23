from dataset_prepare import prepare_data

import keras.backend as K
from keras.utils import np_utils as npu
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from ids import graph_confusion_matrix

def load_model(name):
    # load json and create model
    json_file = open(name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(name+".h5")
    model.compile(loss='mean_squared_error', optimizer='adam')
    print("Loaded model...")
    return model

def multilayer_perceptron():

	scaler = MinMaxScaler(feature_range=(0, 1))
	trainx,trainy,testx,testy,labels = prepare_data()

	testx = scaler.fit_transform(testx)
	temp = testy
	testy = npu.to_categorical(testy)

	model = load_model('idsmodel')

	ypreds = model.predict(testx)

	predictions = np.argmax(ypreds, axis=1)
	attack_corr = 0

	for i in range(len(predictions)):
		if predictions[i] == temp[i]:
			attack_corr += 1
		if predictions[i] == 6:
			predictions[i] = 0
		else:
			predictions[i] = 1

	correct_class = 0
	for i in range(len(labels)):
		if labels[i] == predictions[i]:
			correct_class += 1

	print("\n## Accuracies ##")
	print("Normal/Abnormal activity accuracy:{}%".format((correct_class/len(labels))*100))
	print("Attack type accuracy:{}%".format((attack_corr/len(labels))*100))

	confusion_matrix_metric = confusion_matrix(labels, predictions)
	print("\n## Confusion Matrix ##")
	print("Confusion Matrix for test dataset.")
	print(confusion_matrix_metric)

	graph_confusion_matrix(confusion_matrix_metric)

	# MAIN
if __name__ == '__main__':
	multilayer_perceptron()