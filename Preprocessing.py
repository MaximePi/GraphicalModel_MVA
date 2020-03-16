from Graphe import create_graph
from train import extract_features
#from Transformations import transformations
import os
import numpy as np
    
path = os.getcwd()
db_path = path + '\\MSRC_ObjCategImageDatabase_v2'  

images_name = []
for classe in range(1,20):
    nums = [3,5,10]
    for num in nums:
        name = str(classe)+'_'+str(num)+'_'+'s.bmp'
        images_name.append(name)

print('Segmentation')
#transformations(db_path)      
print('Creating Graphs')
create_graph(db_path, images_name)
print('Extracting Features')
extract_features(db_path,images_name)


    
