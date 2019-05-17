# NIDS_Project_CE417
Networking IDS , network traffic classification with Machine Learning. Final Project for ECE-417.

## Dataset
We are using a dataset with mixed normal and abnormal network traffic by Australian Cybersecurity Center: 
<a href='https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/'> UNSW-NB15_dataset_link </a>
<br>
The dataset is already in the repository in training.csv and testing.csv so there is no need to download from source.
## Prerequisites
Install feature-selector utility library from this github repository: <a href='https://github.com/WillKoehrsen/feature-selector'> Feature_Selector </a>. Please follow the setup instructions there.
<br><br>
Install the other libraries used by the project by entering in console the following command:
```
pip3 install pandas tensorflow matplotlib keras scikit-learn
```
## Run
The IDS can classify incoming packets using Support-Vectors Machine, Random Forest, and Multilayer Perceptron Neural Network.<br>
The default is MLP, so to run with <b>MLP</b> just enter in console:
```
python3 ids.py
```
In order to run using <b>SVM</b>, please enter in console:
```
python3 ids.py svm
```
Lastly, in order to run using <b>Random Forest</b>, please enter in console:
```
python3 ids.py randomforest
```
There is also a mode that runs all three algorithms consecutively, and produces a bar plot to compare the algorithms' results. Please enter in console:
```
python3 ids.py --comparative
```

## Pre-trained Model
Inside the repository there is the idsmodel.h5 and idsmodel.json files which are the pre-trained weights and model that offer 84% accuracy. To test them, please enter in console:
```
python3 test_suite_MLP.py
```

## Authors
* **Evangelou Sotiris** - *Developer* - [Github](https://github.com/EvangelouSotiris)
* **Chatziefremidis Leuteris** - *Developer* - [Github](https://github.com/echatzief)
* **Kalais Konstantinos** - *Developer* - [Github](https://github.com/Kkalais)
