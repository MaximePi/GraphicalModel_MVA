from Graphe import create_graph
from train import extract_features_labels
#from Transformations import transformations
import os
    
path = os.getcwd()
db_path = path + '/MSRC_ObjCategImageDatabase_v2'  

#print('Segmentation')
#transformations(db_path)      
print('Creating Graphs')
create_graph(db_path)
print('Extracting Features')
_ = extract_features_labels(db_path)

    
