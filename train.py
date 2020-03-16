import os
import pickle
import numpy as np
from Adaboost_cl import boosted_tree
from Graphe import create_segmentation_graph
from math import exp

def binarize_label(labels,m):
    ## assign 1 to label m, -1 to other labels  
    labels = np.array(labels)
    bin_labels = np.copy(labels)
 
    bin_labels[bin_labels!=m]=-1
    bin_labels[bin_labels==m]=1

    return bin_labels

def extract_features(db_path):
    
    with open(os.getcwd()+'/color_to_label'+'.pickle', 'rb') as handle: # load decomposed image
        color_to_label = pickle.load(handle)
    
    images_name = os.listdir(db_path+'/Images')
    images_name = [im for im in images_name if im!='Thumbs.db']
    #images_name = images_name[250:301]
    #images_name = ['2_29_s.bmp','15_3_s.bmp','18_21_s.bmp'] # remove to process all images

    Features_leaf, Features_parent = [],[]
    labels_leaf, labels_parent = [],[]
    
    for image_name in images_name:
        print(image_name)
        graph_path = db_path+'/FSG_graphs'
        with open(graph_path+'/'+image_name.split('.')[0]+'.pickle', 'rb') as handle: # load graph with labels
            G = pickle.load(handle)
        create_segmentation_graph(G,graph_path+'/'+image_name.split('.')[0]+'.pickle') # compute features on superpixels
        
        
def infer_labels(graph, color_to_label, leaf_likelihood,parents_likelihood):
    ## Input
    ## FSG graph with features of the image to predict
    p_Y_X = {}
    mask2leaf_id = {}
    
    #compute probabilities
    for leaf in graph.leaf_vertices:
        
        if str(leaf.mask) not in mask2leaf_id.keys():
            mask2leaf_id[str(leaf.mask)] = leaf.get_id()
        
        leaf_id = mask2leaf_id[str(leaf.mask)]
        all_potential_parents = [parent for parent in graph.parent_vertices if graph.get_edges()[leaf.id,parent.id] == 1]
        parent = all_potential_parents[0]
        
        for Y in parents_likelihood.keys():
            for label in parents_likelihood.keys():
                
                new_class1 = np.array([0 for i in parents_likelihood.keys()])
                new_class1[Y] = 1
                leaf.feature.class1 = new_class1
                proba_leaf_label = predict_leaf(Y, leaf.get_features(),color_to_label,leaf_likelihood)
                proba_parent_label = predict_parent(label, parent.get_features(),color_to_label,parents_likelihood)
                
            proba_tot = proba_parent_label*proba_leaf_label
            if leaf_id in p_Y_X.keys():
                if proba_tot in p_Y_X[leaf_id]:
                    p_Y_X[leaf_id][Y] += proba_tot
                else:
                    p_Y_X[leaf_id][Y] = proba_tot
            else:
                p_Y_X[leaf_id] = {Y : proba_tot}
    
    #assemble
    for leaf in graph.leaf_vertices:
        leaf_id = mask2leaf_id[str(leaf.mask)]
        label = max(p_Y_X[leaf_id], key=p_Y_X[leaf_id].get)
        leaf.set_label(label)
    
    return graph
    
        

def train_parents_likelihood_dist(Features_parent,labels_parent,nb_weak_learners=5):    
    ## Output : parent_likelihood dict( key : label | value : [list of K trained trees, matrix of f left/right shape (K,2)]  
    
    parents_likelihood = {}
    for label in np.unique(labels_parent):
        bin_labels = binarize_label(labels_parent,label)
        trees = boosted_tree(Features_parent,bin_labels,nb_weak_learners)
        parents_likelihood[label] = trees
    
    return parents_likelihood

def train_leaf_likelihood_dist(Features_leaf,labels_leaf, nb_weak_learners=5):
    ## Output : leaf_likelihood dict( key : label | value : [list of K trained trees, matrix of f left/right shape (K,2)] 
    
#    leaf_likelihood = {}
#    for label_parent in np.unique(labels_parent):
#        for label_leaf in np.unique(labels_leaf):
#            Features_leaf = Features_leaf_per_parent_label[label_parent]
#            bin_labels = binarize_label(labels_leaf,label_leaf)
#            trees = boosted_tree(Features_leaf,bin_labels,nb_weak_learners)
#            key = str(label_parent)+'_'+str(label_leaf)
#            leaf_likelihood[key] = trees

    leaf_likelihood = {}
    for label in np.unique(labels_leaf):
        bin_labels = binarize_label(labels_parent,label)
        trees = boosted_tree(Features_leaf,bin_labels,nb_weak_learners)
        leaf_likelihood[label] = trees
        
    return leaf_likelihood

def predict_parent(label, parent_feature,color_to_label,parents_likelihood):
    # Input :
    # parent_feature : 1-D array of low level features for the parent
    # color_to_label : dictionnary mapping color to class label
    # parent_likelihood : fitted boosted trees and ratio
    # Output:
    # C dimensional vector of probabilities over the classes
    
        trees,weights = parents_likelihood[label]
        p_label = 0
        for num_learner,tree in enumerate(trees):
            if parent_feature[tree.tree_.feature[0]]<=tree.tree_.treshold:
                p_label+=weights[num_learner,0]
            else:
                p_label+=weights[num_learner,1]
        return np.exp(p_label)

def predict_leaf(label, leaf_feature,color_to_label,leaf_likelihood):
    # Input :
    # leaf_feature : 1-D array of low+high level+class features
    # color_to_label : dictionnary mapping color to class label
    # leaf_likelihood : fitted boosted trees and ratio
    # Output:
    # C dimensional vector of probabilities over the classes
    
    trees,weights = leaf_likelihood[label]
    p_label = 0
    for num_learner,tree in enumerate(trees):
        if leaf_feature[tree.tree_.feature[0]]<=tree.tree_.treshold:
            p_label+=weights[num_learner,0]
        else:
            p_label+=weights[num_learner,1]
    return np.exp(p_label)

def create_matrices(db_path,training_name):
    
    with open(os.getcwd()+'/color_to_label'+'.pickle', 'rb') as handle: # load decomposed image
        color_to_label = pickle.load(handle)
    
    images_name = os.listdir(db_path+'/Images')
    images_name = [image_name for image_name in images_name if image_name in training_name]
    #images_name = ['2_29_s.bmp','15_3_s.bmp','18_21_s.bmp'] # remove to process all images

    Features_leaf, Features_parent = [],[]
    labels_leaf, labels_parent = [],[]
    
    for image_name in images_name:
        print(image_name)
        graph_path = db_path+'/FSG_graphs'
        
        with open(graph_path+'/'+image_name.split('.')[0]+'.pickle', 'rb') as handle: # load graph with labels
            G = pickle.load(handle)
             
        for leaf in G.leaf_vertices:
            Features_leaf.append(leaf.get_features())  ## features shape (num_classes,2*num_low_features)
            labels_leaf.append(leaf.label)
        for parent in G.parent_vertices:
            Features_parent.append(parent.get_features())
            labels_parent.append(parent.label)
                
    Features_leaf = np.stack(Features_leaf)
    Features_parent = np.stack(Features_parent)
    
    
    with open(os.getcwd()+'/Data/Training_Features_leaf'+'.pickle', 'wb') as handle:  # save rgb to label dic
        pickle.dump(Features_leaf_per_parent_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.getcwd()+'/Data/Training_Features_parents'+'.pickle', 'wb') as handle:  # save rgb to label dic
        pickle.dump(Features_parent, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.getcwd()+'/Data/Training_labels_leaf'+'.pickle', 'wb') as handle:  # save rgb to label dic
        pickle.dump(labels_leaf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.getcwd()+'/Data/Training_labels_parents'+'.pickle', 'wb') as handle:  # save rgb to label dic
        pickle.dump(labels_parent, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return Features_leaf_per_parent_label, Features_parent, labels_leaf, labels_parent

def create_train_test_split(db_path):

    images_name = os.listdir(db_path+'/Images')
    images_name = [im for im in images_name if im!='Thumbs.db']
    train_name,test_name = [],[]
    
    for classe in range(1,20):
        train_labels = np.random.randint(1,31,25)
        #test_labels = [ i for i in range(1,31) if not in train_labels]
        for train_label in train_labels:
            name = str(classe)+'_'+str(train_label)+'_'+'s.bmp'
            train_name.append(name)
        for test_label in test_labels:
            name = str(classe)+'_'+str(test_label)+'_'+'s.bmp'
            test_name.append(name)
            
    classe = 20
    train_labels = np.random.randint(1,22,25)
    #test_labels = [ i for i in range(1,22) if not in train_labels]
    for train_label in train_labels:
        name = str(classe)+'_'+str(train_label)+'_'+'s.bmp'
        train_name.append(name)
    for test_label in test_labels:
        name = str(classe)+'_'+str(test_label)+'_'+'s.bmp'
        test_name.append(name)
        
    with open(os.getcwd()+'/training_names'+'.pickle', 'wb') as handle:  # save rgb to label dic
        pickle.dump(train_name, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.getcwd()+'/Data/test_names'+'.pickle', 'wb') as handle:  # save rgb to label dic
        pickle.dump(test_name, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
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

