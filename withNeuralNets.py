import sys
import tensorflow as tf
import numpy as np
from dataset_prepare import prepare_data
from sklearn.preprocessing import MinMaxScaler


def inputTransform(tempX,column_names):
    i=0
    feature_dict = {}
    for feature in column_names:
        feature_dict[feature] = tempX[:,i]
        i=i+1

    print(feature_dict)
    return tf.estimator.inputs.numpy_input_fn(
        x = feature_dict,
        y = None,
        num_epochs = 1,
        batch_size = 1,
        shuffle = True
    )


#Adjust the train and test inputs
def preprocessInputs(train_x,train_y,test_x,test_y,column_names):
    i=0
    feature_dict_train = {}
    feature_dict_test = {}
    for feature in column_names:
        feature_dict_train[feature] = train_x[:,i]
        feature_dict_test[feature] = test_x[:,i]
        i=i+1

    train_input = tf.estimator.inputs.numpy_input_fn(
        x = feature_dict_train,
        y = train_y,
        num_epochs = 1000,
        batch_size = 1,
        shuffle = True
    )

    evaluateInputFunction = tf.estimator.inputs.numpy_input_fn(
        x = feature_dict_test,
        y = test_y,
        num_epochs = 1,
        batch_size=1,
        shuffle=True
    )
    return train_input,evaluateInputFunction
def tweakingParameters(feature_columns,train_input,test_input):
    layerss = [[50,10],[50,20],[80,30],[80,50],[100,80]]
    learn_rate = [0.0001,0.001,0.01,0.1]

    best_lay = None
    best_lr = None
    best_acc = 0
    for h in layerss:
        for l in learn_rate:
            #Prepare the optimizer 
            optimizer_adam = tf.train.AdamOptimizer(learning_rate =l)

            #Train and Evaluate the model
            dnnClassifierModel = tf.estimator.DNNClassifier(hidden_units=h, feature_columns=feature_columns,  optimizer=optimizer_adam)
            dnnClassifierModel.train(input_fn=train_input,steps=10000)
            eval_results = dnnClassifierModel.evaluate(input_fn=test_input, steps=100)

            if(best_acc < eval_results['accuracy']):
                best_acc = eval_results['accuracy']
                best_lr = l
                best_lay = h
    print("Accuracy: ",best_acc*100," %")
    print("Best Layer: ",best_lay)
    print("Best rate: ",best_lr)
#Preprocessing the data with the library we created
train_x,train_y,test_x,test_y,column_names = prepare_data()

#Scale the input data
scaler = MinMaxScaler(feature_range=(0, 1))
train_x = scaler.fit_transform(train_x) 
test_x = scaler.fit_transform(test_x)

## Create the model

# Create the features column
feature_columns = [tf.feature_column.numeric_column(k) for k in column_names]

#Preprocess the inputs
train_input,test_input = preprocessInputs(train_x,train_y,test_x,test_y,column_names)

#Tweaking parameters
#tweakingParameters(feature_columns,train_input,test_input)

#Prepare the optimizer 
optimizer_adam = tf.train.AdamOptimizer(learning_rate =0.01)

#Train and Evaluate the model
dnnClassifierModel = tf.estimator.DNNClassifier(hidden_units=[50,10], feature_columns=feature_columns,  optimizer=optimizer_adam)
dnnClassifierModel.train(input_fn=train_input,steps=10000)
eval_results = dnnClassifierModel.evaluate(input_fn=test_input, steps=100)
print(eval_results)

#Test for the first row
sl = np.array([test_x[0,:]])
transformed = inputTransform(sl,column_names)
predictions = dnnClassifierModel.predict(input_fn=transformed)
out = test_y[0]

for x in predictions:
    print(x)
    print("Class: ",out)