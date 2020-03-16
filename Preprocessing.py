from Graphe import create_graph
from train import extract_features
#from Transformations import transformations
import os

path = os.getcwd()
db_path = path + '\\MSRC_ObjCategImageDatabase_v2'  

print('Segmentation')
#transformations(db_path)      
print('Creating Graphs')
create_graph(db_path)
print('Extracting Features')
extract_features(db_path)

from testing_routine import *
from training_routine import *

path = os.getcwd()
db_path = path + '/MSRC_ObjCategImageDatabase_v2'  

#create_train_test_split(db_path) # Deja fait
with open(os.getcwd()+'/Data/train_names'+'.pickle', 'rb') as handle:  # save rgb to label dic
    training_names = pickle.load(handle)
with open(os.getcwd()+'/color_to_label'+'.pickle', 'rb') as handle:  # save rgb to label dic
    color_to_label = pickle.load(handle)

N_labels = len(color_to_label)
Features_leaf, Features_parent, labels_leaf, labels_parent = create_matrices(db_path,training_names,N_labels)
print('parent likelihood')
parents_likelihood = train_parents_likelihood_dist(Features_parent,labels_parent,nb_weak_learners=5)
with open(os.getcwd()+'/Model/parents_likelihood_debug'+'.pickle', 'wb') as handle:  # save rgb to label dic
    pickle.dump(parents_likelihood, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print('leaf likelihood')
leaf_likelihood = train_leaf_likelihood_dist(Features_leaf,labels_leaf,nb_weak_learners=5)
with open(os.getcwd()+'/Model/leaf_likelihood_debug'+'.pickle', 'wb') as handle:  # save rgb to label dic
    pickle.dump(leaf_likelihood, handle, protocol=pickle.HIGHEST_PROTOCOL)

    

path = os.getcwd()
db_path = path + '/MSRC_ObjCategImageDatabase_v2'  

with open(os.getcwd()+'/Model/parents_likelihood_debug.pickle', 'rb') as handle:  # save rgb to label dic
    parents_likelihood =  pickle.load(handle)
with open(os.getcwd()+'/Model/leaf_likelihood_debug.pickle', 'rb') as handle:  # save rgb to label dic
    leaf_likelihood =  pickle.load(handle)
with open(os.getcwd()+'/Data/train_names'+'.pickle', 'rb') as handle:  # save rgb to label dic
    testing_names = pickle.load(handle)
with open(os.getcwd()+'/color_to_label'+'.pickle', 'rb') as handle:  # save rgb to label dic
    color_to_label = pickle.load(handle)

N_labels = len(color_to_label)

test(db_path,testing_names,N_labels, color_to_label,leaf_likelihood,parents_likelihood)
    


    
