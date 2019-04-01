from dataset_prepare import prepare_data
from sklearn.svm import SVC
import sys

train_x,train_y,test_x,test_y = prepare_data()

if len(sys.argv) == 2:
	C = float(sys.argv[1])
if len(sys.argv) == 3:
	C = float(sys.argv[1])
	gamma = float(sys.argv[2])
else:
	C=1
	gamma =100

train_y = train_y.reshape((train_y.shape[0],))


print(C,gamma)
classifier = SVC(kernel="rbf",C=C,gamma=gamma, verbose=True)
classifier.fit(train_x[:50000,:], train_y[:50000])
SVs = classifier.support_vectors_ #support vectors

print("For C : ",C," Gamma: ",gamma) 
print("Number of Support Vectors: %d" %len(SVs))
print('\n')
print("Accuracy: {}%".format(classifier.score(test_x[:1000,:], test_y[:1000]) * 100 ))
