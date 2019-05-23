import pandas as pd 
import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from feature_selector import FeatureSelector

def clean_nominals_and_create_our_datasets(train_set_df,test_set_df):

	#Get nominal columns
	nominal_cols = train_set_df.select_dtypes(include='object').columns.tolist()

	#Turn nominal to numeric train and testX
	for nom in nominal_cols:
	    le = LabelEncoder()
	    le.fit(train_set_df[nom])
	    train_set_df[nom]=le.transform(train_set_df[nom])
	    testEnc = LabelEncoder()
	    testEnc.classes_=le.classes_
	    testEnc.fit(test_set_df[nom])
	    test_set_df[nom] =testEnc.transform(test_set_df[nom])

	# drop the nominal columns from the initial set
	train_set_df_y = train_set_df.attack_cat
	train_Y = np.array(train_set_df_y)
	train_Y = train_Y.reshape((train_Y.shape[0],1))
	train_set_df = train_set_df.drop(["attack_cat","label"], axis=1)
	train_X = np.array(train_set_df)

	test_set_df_y = test_set_df.attack_cat
	test_set_df_y1 = test_set_df.label
	test_Y = np.array(test_set_df_y)
	test_Y = test_Y.reshape((test_Y.shape[0],1))
	test_set_df = test_set_df.drop(["attack_cat","label"], axis=1)
	test_X = np.array(test_set_df)

	labels = np.array(test_set_df_y1)
	labels = labels.reshape((labels.shape[0],1))
	
	return train_X,train_Y,test_X,test_Y,labels

def prepare_data(): 
	"""
	This function is the main of this module. calls the above functions in order to read/clean/save
	our data in usable form.
	I created this function to use dataset_prepare.py as a Python module in our main program.
	
	Return values: training X,Y dataset and testing X,Y dataset
	"""

	# read our csv files
	features_df = pd.read_csv("UNSW_NB15_features.csv",encoding = "ISO-8859-1")
	training_df = pd.read_csv("training.csv").drop("id",axis=1)
	testing_df = pd.read_csv("testing.csv").drop("id",axis=1)

	fs = FeatureSelector(data = training_df)
	fs.identify_collinear(correlation_threshold=0.85)
	training_df = fs.remove(methods = ['collinear'],keep_one_hot = True)
	columnlist = list(training_df)
	testing_df = testing_df[columnlist]
	
	training_df = training_df.sample(frac=1)
	testing_df = testing_df.sample(frac=1)
	train_x,train_y,test_x,test_y, labels = clean_nominals_and_create_our_datasets(training_df,testing_df)

	training_df = training_df.drop(["attack_cat","label"], axis=1)
	print("The features we will use are: ", np.array(list(training_df)))

	return train_x,train_y,test_x,test_y,labels
