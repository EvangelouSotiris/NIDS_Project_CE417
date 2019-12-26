import pandas as pd 
import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from feature_selector import FeatureSelector

def retrieve_classes(train_set_df):
    # Get nominal columns
    nominal_cols = train_set_df.select_dtypes(include='object').columns.tolist()
            
    label_classes = {}
    
    # Turn nominal to numeric train
    for nom in nominal_cols:
        le = LabelEncoder()
        le.fit(train_set_df[nom])
        label_classes[nom] = le.classes_
    
    return label_classes,nominal_cols

def transform_to_nominal(): 
    # read our csv files
    training_df = pd.read_csv("training.csv").drop("id",axis=1)
    
    # Feature selector
    fs = FeatureSelector(data = training_df)
    fs.identify_collinear(correlation_threshold=0.85)
    training_df = fs.remove(methods = ['collinear'],keep_one_hot = True)

    training_df = training_df.sample(frac=1)
    training_df = training_df.drop(["attack_cat","label"], axis=1)
    columnList = list(training_df)
    labels,nominal_cols = retrieve_classes(training_df)
    
    return labels,nominal_cols,columnList
