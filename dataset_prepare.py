import pandas as pd 
import numpy as np
from scipy.stats.stats import pearsonr

def get_all_nominals_in_list(real_data):
	'''
	For example: data.protocol = [udp,udp,tcp,igmp,tcp,ipv6] returns [udp,tcp,igmp,ipv6]
	we will use the position of each unique nominal to mark a nominal with an int
	'''
	unique_nominals = []
	for nominal in real_data:
		if nominal not in unique_nominals:
			unique_nominals.append(nominal)
	return unique_nominals

def find_unique_nominal_pos(nominal, unique_nominals):
	for i in range(len(unique_nominals)):
		if unique_nominals[i] == nominal:
			return i
	exit("ERROR")

def substitute_nominal(unique_nominals, column):
	new_column = []
	for item in column:
		sub_integer = find_unique_nominal_pos(item,unique_nominals)
		new_column.append(sub_integer)
	return new_column

def clean_nominals_and_create_our_datasets(training_set_df):
	# get every unique nominal in an array so we can then substitute it with an int
	unique_protocols = get_all_nominals_in_list(training_set_df.proto)
	unique_states = get_all_nominals_in_list(training_set_df.state)
	unique_services = get_all_nominals_in_list(training_set_df.service)

	# create substituted to integer columns
	proto_column = substitute_nominal(unique_protocols, np.array(training_set_df.proto))
	state_column = substitute_nominal(unique_states, np.array(training_set_df.state))
	service_column = substitute_nominal(unique_services, np.array(training_set_df.service))

	# drop the nominal columns from the initial set
	training_set_df_y = training_set_df.label
	train_Y = np.array(training_set_df_y)
	train_Y = train_Y.reshape((train_Y.shape[0],1))
	training_set_df = training_set_df.drop(["proto","state","service","attack_cat","label"], axis=1)
	training_set = np.array(training_set_df)
	training_set = np.c_[training_set,proto_column]
	training_set = np.c_[training_set,state_column]
	train_X = np.c_[training_set,service_column]

	return train_X,train_Y

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

	train_x,train_y = clean_nominals_and_create_our_datasets(training_df)
	test_x,test_y = clean_nominals_and_create_our_datasets(testing_df)

	train_x,test_x = delete_higly_correlated_cols(train_x,test_x)

	## Just to show that the data given need some work to use
	feature_list = np.array(features_df.Name)
	features_of_training_set = np.array(list(training_df))
	print("Features mentioned in the features.csv but are not included in the dataset:")
	for i in range(len(feature_list)):
		if feature_list[i] not in features_of_training_set:
			print(feature_list[i])

	training_df = training_df.drop(["proto","state","service","attack_cat","label","ct_ftp_cmd"], axis=1)
	print("The features we will use are: ", np.array(list(training_df)))

	
	
	return train_x,train_y,test_x,test_y

prepare_data()
# TODO : LIKE THE REST OF THE PROGRAM. I THINK THE DATA PREPROCESSING WORK IS DONE.