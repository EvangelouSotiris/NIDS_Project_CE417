import os

while True:
  #os.system("ra -r - -s dur proto ")
  os.system("ra -r - -A -L0 -c , -s dur proto state spkts dpkts rate sttl dttl sload dload load sjit swin stcpb dtcpb tcprtt smeansz dmeansz")
  #os.system("ra -r - -A -s *")
