from sklearn.ensemble import RandomForestClassifier
from dataset_prepare import prepare_data

#Prepare the data
train_x,train_y,test_x,test_y ,axristo= prepare_data()
train_y = train_y.reshape((train_y.shape[0],))

#Initialize the classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
clf.fit(train_x,train_y)

#Test the model
print("Accuracy: {}%".format(clf.score(test_x,test_y)*100))