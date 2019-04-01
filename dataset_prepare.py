import pandas as pd 
import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import LabelEncoder

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
	train_set_df_y = train_set_df.label
	train_Y = np.array(train_set_df_y)
	train_Y = train_Y.reshape((train_Y.shape[0],1))
	train_set_df = train_set_df.drop(["attack_cat","label"], axis=1)
	train_X = np.array(train_set_df)

	test_set_df_y = test_set_df.label
	test_Y = np.array(test_set_df_y)
	test_Y = test_Y.reshape((test_Y.shape[0],1))
	test_set_df = test_set_df.drop(["attack_cat","label"], axis=1)
	test_X = np.array(test_set_df)
	
	return train_X,train_Y,test_X,test_Y

def delete_higly_correlated_cols(train_x, test_x):
	tuple_correlated = []
	for i in range(train_x.shape[1]):
		for j in range(train_x.shape[1]):
			if i != j and abs(pearsonr(train_x[:,i],train_x[:,j])[0]) == 1:
				if (j,i) not in tuple_correlated:
					tuple_correlated.append((i,j)) 
	# columns 1,3,10 - 2,4,11 - 12,38 - 16,19 - 20,21 - 20,22 - 27,32,37 - 29,30 - 30,31 - 33,34
	# are highly correlated but we need them for the abnormal 1% case
	# but we toss one of the two columns that are 100% correlated -> 33-34 -> toss 34
	for tp in tuple_correlated:
		i,j = tp
		train_x = np.delete(train_x,j,1) # delete jth column that is 100% correlated with ith col
		test_x = np.delete(test_x,j,1)

	return train_x,test_x

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

	train_x,train_y,test_x,test_y = clean_nominals_and_create_our_datasets(training_df,testing_df)

	train_x,test_x = delete_higly_correlated_cols(train_x,test_x)

	training_df = training_df.drop(["attack_cat","label","ct_ftp_cmd"], axis=1)
	print("The features we will use are: ", np.array(list(training_df)))

	return train_x,train_y,test_x,test_y

prepare_data()
# TODO : LIKE THE REST OF THE PROGRAM. I THINK THE DATA PREPROCESSING WORK IS DONE.