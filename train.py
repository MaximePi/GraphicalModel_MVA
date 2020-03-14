import os
import pickle
import numpy as np
from Adaboost_cl import boosted_tree
from Graphe import create_segmentation_graph

def binarize_label(labels,m):
    ## assign 1 to label m, -1 to other labels  
    labels = np.array(labels)
    bin_labels = np.copy(labels)
 
    bin_labels[bin_labels!=m]=-1
    bin_labels[bin_labels==m]=1

    return bin_labels

def extract_features_labels(db_path):
    ## Compute features from raw graphs for each images
    ## Save graphs with features 
    ## Save classification dataset and labels 
    
    ### Output
    ## label leaf , label for all the leaf of all the FSG
    ## label parent , label for all the leaf of all the FSG
    ## Features_leaf_per_parent_label, dict | key = label of parent | value = low feat of leaf+ avg high feat of parents with key label -> shape (num total leaf,2*num_low_features)
    ## Features_parent, shape (num total parent,num_low_features )
    
    with open(os.getcwd()+'/color_to_label'+'.pickle', 'rb') as handle: # load decomposed image
        color_to_label = pickle.load(handle)
    
    images_name = os.listdir(db_path+'/Images')
    images_name = ['2_29_s.bmp','15_3_s.bmp','18_21_s.bmp'] # remove to process all images

    Features_leaf, Features_parent = [],[]
    labels_leaf, labels_parent = [],[]
    
    for image_name in images_name:
        print(image_name)
        graph_path = db_path+'/FSG_graphs'
        
        with open(graph_path+'/'+image_name.split('.')[0]+'.pickle', 'rb') as handle: # load graph with labels
            G = pickle.load(handle)
        G = create_segmentation_graph(G,color_to_label) # compute features on superpixels
        
        with open(graph_path+'/'+image_name.split('.')[0]+'.pickle', 'wb') as handle: # save Graph
            pickle.dump(G, handle, protocol=pickle.HIGHEST_PROTOCOL)  
            
        for leaf in G.leaf_vertices:
            Features_leaf.append(leaf.get_features())  ## features shape (num_classes,2*num_low_features)
            labels_leaf.append(leaf.label)
        for parent in G.parent_vertices:
            Features_parent.append(parent.get_features())
            labels_parent.append(parent.label)
                

    Features_leaf_per_parent_label = {}
    for j in np.unique(labels_parent):
        Features_leaf_per_parent_label[j] = np.stack([feat[j,:] for feat in Features_leaf])
    Features_parent = np.stack(Features_parent)
    
    
    with open(os.getcwd()+'/Data/Feat_leaf'+'.pickle', 'wb') as handle:  # save rgb to label dic
        pickle.dump(Features_leaf_per_parent_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.getcwd()+'/Data/Feat_parents'+'.pickle', 'wb') as handle:  # save rgb to label dic
        pickle.dump(Features_parent, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.getcwd()+'/Data/lab_leaf'+'.pickle', 'wb') as handle:  # save rgb to label dic
        pickle.dump(labels_leaf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.getcwd()+'/Data/lab_parents'+'.pickle', 'wb') as handle:  # save rgb to label dic
        pickle.dump(labels_parent, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return Features_leaf_per_parent_label, Features_parent, labels_leaf, labels_parent

def train_parents_likelihood_dist(Features_parent,labels_parent,nb_weak_learners=5):    
    ## Output : parent_likelihood dict( key : label | value : [list of K trained trees, matrix of f left/right shape (K,2)]  
    
    parents_likelihood = {}
    for label in np.unique(labels_parent):
        bin_labels = binarize_label(labels_parent,label)
        trees = boosted_tree(Features_parent,bin_labels,nb_weak_learners)
        parents_likelihood[label] = trees
    
    return parents_likelihood

def train_leaf_likelihood_dist(Features_leaf_per_parent_label,labels_leaf,labels_parent, nb_weak_learners=5):
    ## Output : leaf_likelihood dict( key : label parent_ label_leaf | value : [list of K trained trees, matrix of f left/right shape (K,2)]    
    
    leaf_likelihood = {}
    for label_parent in np.unique(labels_parent):
        for label_leaf in np.unique(labels_leaf):
            Features_leaf = Features_leaf_per_parent_label[label_parent]
            bin_labels = binarize_label(labels_leaf,label_leaf)
            trees = boosted_tree(Features_leaf,bin_labels,nb_weak_learners)
            key = str(label_parent)+'_'+str(label_leaf)
            leaf_likelihood[key] = trees
        
    return leaf_likelihood


#path = os.getcwd()
#db_path = path + '/MSRC_ObjCategImageDatabase_v2'
#Features_leaf_per_parent_label, Features_parent, labels_leaf, labels_parent = extract_features_labels(db_path)

### Training 

#parents_likelihood = train_parents_likelihood_dist(Features_parent,labels_parent,nb_weak_learners=5)
#leaf_likelihood = train_leaf_likelihood_dist(Features_leaf_per_parent_label,labels_leaf,labels_parent)
#with open(os.getcwd()+'/Model/parent_likelihood'+'.pickle', 'wb') as handle:  # save rgb to label dic
#        pickle.dump(parents_likelihood, handle, protocol=pickle.HIGHEST_PROTOCOL)
#with open(os.getcwd()+'/Model/leaf_likelihood'+'.pickle', 'wb') as handle:  # save rgb to label dic
#    pickle.dump(leaf_likelihood, handle, protocol=pickle.HIGHEST_PROTOCOL)