import sys
import keras
from keras.models import model_from_json
from onlyLabelsPrepare import transform_to_nominal
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from os import path
import os

#import tensorflow.python.util.deprecation as deprecation
#deprecation._PRINT_DEPRECATION_WARNINGS = False


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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

def ct_state_ttl(sttl,dttl,state):
  ct_state = "0"
  if sttl == ('62' or '63' or '254' or '255') and dttl == ('252' or '253') and state == 'FIN':
    ct_state = "1"
  elif sttl == ('0' or '62' or '254') and dttl == '0' and state =='INT':
    ct_state = "2"
  elif sttl == ('62' or '254') and dttl == ('60' or '252' or '253') and state =='CON':
    ct_state = "3"
  elif sttl == '254' and dttl == '252' and state == 'ACC' :
    ct_state = "4"
  elif sttl == '254' and dttl == '252' and state == 'CLO':
    ct_state = "5"
  elif sttl == '254' and dttl == '0' and state == 'REQ':
    ct_state = "6"
  else:
    ct_state = "0"
  
  return ct_state

def define_service(dstport):
  if dstport == "https" :
    service = "https"
    dstport = "443"
  elif dstport == "http":
    service = "http"
    dstport = "80"
  elif dstport == "ftp":
    service = "ftp"
    dstport = "21"
  elif dstport == "domain":
    service = "domain"
    dstport = "53"
  elif dstport == "mdns":
    service = "dns"
    dsport = "5353"
  elif dstport == "netbios-dgm":
    service = "netbios-dgm"
    dstport = "138"
  elif dstport == "smtp":
    service = "smtp"
    dstport = "25"
  elif dstport == "snmp":
    service = "snmp"
    dstport = "161"
  elif dstport == "ftp-data":
    service = "ftp-data"
    dstport = "20"
  elif dstport == "ssh":
    service = "ssh"
    dstport = "22"
  elif dstport == "dhcp":
    service = "dhcp"
    dstport = "546"
  elif dstport == "irc":
    service = "irc"
    dstport = "194"
  elif dstport == "pop3":
    service = "pop3"
    dstport = "995"
  elif dstport == "radius":
    service = "radius"
    dstport = "1812"
  elif dstport == "ssl":
    service = "ssl"
    dstport = "443" 
  else :
    service = "-"

  return dstport,service



def is_ftp_login(service):
  if service == 'ftp':
    return 1
  else: 
    return 0

def basic_tokens(line):
  tokens = line.split(",")
  # Retrieve the service
  srcip = tokens[3]
  srcport = tokens[4]
  dstip = tokens[6]
  dstport = tokens[7]
  res_bd_len = tokens[9]
  ltime = tokens[12]
  
  # Retrieve the service 
  dstport,service = define_service(dstport)
 
  if service == "http" or service == "https":
    res_bd_len = tokens[9]
  else:
    res_bd_len = "0"

  # Get the other arguments
  tokens = tokens[13:]
  tokens.insert(2,service)
  
  # Replace '' with 0 and remove \n 
  for i in range(len(tokens)):
    if tokens[i] == "":
      tokens[i] = "0"
    elif "\n" in tokens[i]:
      tokens[i] = tokens[i].replace("\n","")

  # insert ct_state_ttl
  ct_state_t = ct_state_ttl(tokens[7],tokens[8],tokens[3])
  tokens.insert(len(tokens)+1,ct_state_t)
  
  # ftp login
  tokens.insert(len(tokens)+1,is_ftp_login(service))

  return tokens,srcip,srcport,dstip,dstport,ltime,res_bd_len
def ct_srv_src(curIp,curService,nextIp,nextService,ct_srv):
  if curIp == nextIp and curService == nextService:
    ct_srv+=1
  elif curIp != nextIp and curService != nextService:
    ct_srv= 1
  else:
    ct_srv = 0
  return ct_srv
def ct_flw_http_mthd(srcip,dstip,sport,dstport,nextSrcip,nextDstip,nextSport,nextDsport,method,ct_flow):
  if srcip == nextSrcip and dstip == nextDstip and sport == nextSport and dstport == nextDsport and (method!="-" ): 
    ct_flow +=1
  else: 
    ct_flow = 0
  return ct_flow  
def ct_dst_ltm(dstIp,ltime,nextDstIp,nextLtime,ct_dst):
  if dstIp == nextDstIp and ltime == nextLtime:
    ct_dst += 1
  else:
    ct_dst = 1
  return ct_dst    
def multilayer_perceptron():
    
    # Load the model
    model = load_model('idsmodel')
    
    print("Extracting labels...")
    # Prepare data( extract classes in order to turn to nominal ) 
    label_classes,nominal_cols,columns = transform_to_nominal()
    
    print("Features we willl use: \n"+str(columns))
    cols = ['dur','proto','service','state','spkts','dpkts','rate','sttl','dttl','sload','dload','sintpkt','dintpkt','sjit','djit','swin','stcpb','dtcpb','tcprtt','smeansz','dmeansz','trans_depth','response_body_len','ct_srv_src','ct_state_ttl','ct_dst_ltm','is_ftp_login','ct_flw_http_mthd']
   
    print("\n\nStarting the real time NIDS....\n\n")
    # Read the line
    firstLine = input()

    #GLOBAL VARIABLES
    ct_srv = 0
    ct_flow = 0 
    ct_dst = 0
    while True:

      if "Dur" not in firstLine: 
        secLine = input()
        
        # Retrieve the tokens of the two lines
        firstLine_tokens,firstSrcIP,firstSrcPort,firstDstIP,firstDstPort,firstLtime,first_bd_len =basic_tokens(firstLine)
        secLine_tokens,secSrcIP,secSrcPort,secDstIP,secDstPort,secLtime,sec_bd_len = basic_tokens(secLine)
        
        # Response boby length ( We need to fix that shit )
        firstLine_tokens.insert(22,first_bd_len)

        #Check for the ct_srv 
        ct_srv = ct_srv_src(firstSrcIP,firstLine_tokens[2],secSrcIP,secLine_tokens[2],ct_srv)
        firstLine_tokens.insert(23,str(ct_srv))
        
        #Check for the ct_dst_lt 
        ct_dst = ct_dst_ltm(firstDstIP,firstLtime,secDstIP,secLtime,ct_dst)
        firstLine_tokens.insert(25,str(ct_dst))        

        #Check ct_flw_http_mthd
        ct_flow = ct_flw_http_mthd(firstSrcIP,firstDstIP,firstSrcPort,firstDstPort,secSrcIP,secDstIP,secSrcPort,secDstPort,firstLine_tokens[2],ct_flow)
        firstLine_tokens.insert(len(firstLine_tokens)+1,str(ct_flow))
        
        firstLine_tokens = np.array([firstLine_tokens])
        df = pd.DataFrame(columns=cols,data=firstLine_tokens)
        keepDF = df.copy()
        
        # Turn from nominal to numeric
        for nom in nominal_cols:

          le = LabelEncoder()
          le.fit(df[nom])
          
          # Check if the classes already included to the encoder
          if le.classes_[0] not in label_classes[nom]:
            le.classes_ = np.concatenate((label_classes[nom],le.classes_))
          else:
            le.classes_ = label_classes[nom]

          # Transform the nominal to numberic
          df[nom] = le.transform(df[nom])

        # String to numbers
        dataForTest = np.array(df)[0]
        dataForTest = np.array([float(it) for it in dataForTest])
        
        # Predict
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataForTest = scaler.fit_transform(dataForTest.reshape(-1,1))

        dataForTest = dataForTest.reshape(1,28)
        ypred = model.predict(dataForTest)
        ypred = np.argmax(ypred,axis=1)

        classes = ['Analysis','Backdoor','DoS','Exploits','Fuzzers','Generic','Normal','Reconnaissance','Shellcode','Worms']

        if ypred[0] == 6:
          print(bcolors.OKGREEN+"Normal Behavior"+bcolors.ENDC)
        else:
          print(bcolors.FAIL+"Possible '" + classes[ypred[0]] + "' Attack : added to out.csv for analysis." + bcolors.ENDC)
          df = pd.DataFrame(dataForTest)
          df.insert(df.shape[1]-1,column='predclass',value=classes[ypred[0]])
          if path.exists('out.csv'):
            df.to_csv('out.csv',mode='a', header=False)
          else:
            df.to_csv('out.csv')
        #predictions = np.argmax(ypreds, axis=1)
        firstLine = secLine
      else:
        secLine = input() 
        firstLine = secLine

if __name__ == "__main__":
  multilayer_perceptron()
