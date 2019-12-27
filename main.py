import sys
from keras.models import model_from_json
from onlyLabelsPrepare import transform_to_nominal
from sklearn.preprocessing import LabelEncoder
import pandas as pd
def load_model(name):
    #load json and create model
    json_file = open(name+".json","r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(name+".h5")
    model.compile(loss='mean_squared_error', optimizer='adam')
    print("Loaded model...")
    return model

def multilayer_perceptron():
    
    # Load the model
    model = load_model('idsmodel')

    print("Extracting labels...")
    # Prepare data( extract classes in order to turn to nominal ) 
    label_classes,nominal_cols,columns = transform_to_nominal()
    
    print(columns)
    cols = ['dur','proto','state','spkts','dpkts','rate','sttl','dttl','sload','dload','sjit','djit','swin','stcpb','dtcpb','tcprtt','smeansz','dmeansz']
    print(len(cols))
    # Read the line
    for line in sys.stdin:
        if "Dur" not in line:
            tokens = line.split(",")
            print(tokens)
            #df = pd.DataFrame()
            #for i in range(len(cols)):
                #df[cols[i]] = tokens[i]
            #print(df)
            #df = pd.DataFrame(tokens)
def test():
    for line in sys.stdin:
        #if "Dur" not in line
        print(line)

if __name__ == "__main__":
    #test()
    multilayer_perceptron()
