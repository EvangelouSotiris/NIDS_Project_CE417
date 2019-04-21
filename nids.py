from dataset_prepare import prepare_data
from sklearn.svm import SVC
import sys
import tensorflow as tf

train_x,train_y,test_x,test_y,column_names = prepare_data()

if len(sys.argv) == 2:
	C = float(sys.argv[1])
if len(sys.argv) == 3:
	C = float(sys.argv[1])
	gamma = float(sys.argv[2])
else:
	C=1
	gamma =100

train_y = train_y.reshape((train_y.shape[0],))
"""
print(column_names)
print(train_x.shape)

feature_columns = [tf.feature_column.numeric_column(k) for k in column_names]

feature_dict_train = {}
feature_dict_test = {}

i = 0
for feature in column_names:
	feature_dict_train[feature] = train_x[:,i]
	feature_dict_test[feature] = test_x[:,i]
	i += 1

train_input = tf.estimator.inputs.numpy_input_fn(
	x = feature_dict_train,
	y = train_y,
	num_epochs = None,
	batch_size = 1,
	shuffle = True
)

evaluateInputFunction = tf.estimator.inputs.numpy_input_fn(
	x = feature_dict_test,
	y = test_y,
	num_epochs = 1,
	shuffle = False
)

dnnClassifierModel = tf.estimator.DNNClassifier(hidden_units=[50, 20],
	feature_columns=feature_columns,
	n_classes=2,
	activation_fn=tf.nn.tanh,
	optimizer=lambda: tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(learning_rate=0.001,
		global_step=tf.train.get_global_step(),
		decay_steps=1000,decay_rate=0.96)
	)
)

dnnClassifierModel.train(input_fn=train_input,steps=100)
dnnClassifierModel.evaluate(evaluateInputFunction)
"""
print(C,gamma)
classifier = SVC(kernel="rbf",C=C,gamma=gamma, verbose=True)
classifier.fit(train_x[:50000,:], train_y[:50000])
SVs = classifier.support_vectors_ #support vectors

print("For C : ",C," Gamma: ",gamma) 
print("Number of Support Vectors: %d" %len(SVs))
print('\n')
print("Accuracy: {}%".format(classifier.score(test_x[:1000,:], test_y[:1000]) * 100 ))
