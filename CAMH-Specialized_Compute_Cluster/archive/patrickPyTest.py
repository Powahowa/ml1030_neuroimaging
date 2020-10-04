import os

saveDir = "/external/rprshnas01/netdata_kcni/edlab/temp_dataknights/patrickTesting/"

os.chdir(saveDir)

f = open('helloworld.txt','w')
f.write('Working Directory is:' + os.getcwd())
f.close()