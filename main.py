import sys
import keras
from keras.models import model_from_json
from onlyLabelsPrepare import transform_to_nominal
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

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

def is_ftp_login(service):
  if service == 'ftp':
    return 1
  else: 
    return 0

def basic_tokens(line):
  tokens = line.split(",")
  #print(tokens)
  # Retrieve the service
  srcip = tokens[3]
  srcport = tokens[4]
  dstip = tokens[6]
  dstport = tokens[7]
  ltime = tokens[12]

  if dstport == "https" :
    service = "https"
    dstport = "443"
  elif dstport == "http":
    service = "http"
    dstport = "80"
  elif dstport == "ftp":
    service = "ftp"
    dstport = "21"
  else :
    service = "-"

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

  return tokens,srcip,srcport,dstip,dstport,ltime
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
    
    print(columns)
    cols = ['dur','proto','service','state','spkts','dpkts','rate','sttl','dttl','sload','dload','sintpkt','dintpkt','sjit','djit','swin','stcpb','dtcpb','tcprtt','smeansz','dmeansz','trans_depth','response_body_len','ct_srv_src','ct_state_ttl','ct_dst_ltm','is_ftp_login','ct_flw_http_mthd']
    print(len(cols))
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
        firstLine_tokens,firstSrcIP,firstSrcPort,firstDstIP,firstDstPort,firstLtime =basic_tokens(firstLine)
        secLine_tokens,secSrcIP,secSrcPort,secDstIP,secDstPort,secLtime = basic_tokens(secLine)
        
        
        # Response boby length ( We need to fix that shit )
        firstLine_tokens.insert(22,"0")

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
        print(df)
        firstLine = secLine
      else:
        secLine = input() 
        firstLine = secLine

if __name__ == "__main__":
  multilayer_perceptron()
