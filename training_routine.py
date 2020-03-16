import os
import pickle
import numpy as np
from Adaboost_cl import boosted_tree

def binarize_label(labels,m):
    ## assign 1 to label m, -1 to other labels  
    labels = np.array(labels)
    bin_labels = np.copy(labels)
 
    bin_labels[bin_labels!=m]=-1
    bin_labels[bin_labels==m]=1

    return bin_labels


def create_train_test_split(db_path):

    images_name = os.listdir(db_path+'/Images')
    images_name = [im for im in images_name if im!='Thumbs.db']
    train_name,test_name = [],[]
    
    for classe in range(1,20):
        train_labels = np.arange(1,26)
        #train_labels = np.random.randint(1,31,25)
        #train_labels = np.unique(train_labels)
        #while len(train_labels)<25:
        #    train_labels = np.concatenate((train_labels,np.random.randint(1,31,25-len(train_labels))))
        #    train_labels = np.unique(train_labels)
        test_labels = [ i for i in range(1,31) if i not in train_labels]
        for train_label in train_labels:
            name = str(classe)+'_'+str(train_label)+'_'+'s.bmp'
            train_name.append(name)
        for test_label in test_labels:
            name = str(classe)+'_'+str(test_label)+'_'+'s.bmp'
            test_name.append(name)
            
    classe = 20
    train_labels = np.arange(1,26)
    #train_labels = np.random.randint(1,22,25)
    #train_labels = np.unique(train_labels)
    
    #while len(train_labels)<25:
    #    train_labels = np.concatenate((train_labels,np.random.randint(1,22,25-len(train_labels))))
    #    train_labels = np.unique(train_labels)
    test_labels = [ i for i in range(1,22) if i not in train_labels]
    for train_label in train_labels:
        name = str(classe)+'_'+str(train_label)+'_'+'s.bmp'
        train_name.append(name)
    for test_label in test_labels:
        name = str(classe)+'_'+str(test_label)+'_'+'s.bmp'
        test_name.append(name)
        
    with open(os.getcwd()+'/Data/train_names'+'.pickle', 'wb') as handle:  # save rgb to label dic
        pickle.dump(train_name, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.getcwd()+'/Data/test_names'+'.pickle', 'wb') as handle:  # save rgb to label dic
        pickle.dump(test_name, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
def create_matrices(db_path,training_name,N_labels):

    #images_name = os.listdir(db_path+'/Images')
    #images_name = [image_name for image_name in images_name if image_name in training_name]
    images_name = ['2_28_s.bmp','4_8_s.bmp', '6_15_s.bmp'] # remove to process all images

    Features_leaf, Features_parent = [],[]
    labels_leaf, labels_parent = [],[]
    
    for image_name in images_name:
        print(image_name)
        graph_path = db_path+'/FSG_graphs_final'
        
        with open(graph_path+'/'+image_name.split('.')[0]+'.pickle', 'rb') as handle: # load graph with labels
            G = pickle.load(handle)
             
        for leaf in G.leaf_vertices:
            ll = leaf.get_features(N_labels)
            if len(ll) > 3440: ## leaf w. no parents
                Features_leaf.append(ll)  ## features shape (num_classes,2*num_low_features)
                labels_leaf.append(leaf.label)
            
        for parent in G.parent_vertices:
            par_feat = parent.get_features(N_labels)
            if par_feat not in Features_parent: # bc duplicates among parents | one leaf, one parent
                Features_parent.append(par_feat)
                labels_parent.append(parent.label)
                
    #max_len = max([len(feat) for feat in Features_leaf])
    #padded_features = []
    #for feat in Features_leaf:
    #    padded_features.append(np.concatenate((feat,[0]*(max_len-len(feat)))))

    Features_leaf = np.stack(Features_leaf)
    Features_parent = np.stack(Features_parent)
    print(Features_leaf.shape)
    print(Features_parent.shape)
    with open(os.getcwd()+'/Data/Training_Features_leaf'+'.pickle', 'wb') as handle:  # save rgb to label dic
        pickle.dump(Features_leaf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.getcwd()+'/Data/Training_Features_parents'+'.pickle', 'wb') as handle:  # save rgb to label dic
        pickle.dump(Features_parent, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.getcwd()+'/Data/Training_labels_leaf'+'.pickle', 'wb') as handle:  # save rgb to label dic
        pickle.dump(labels_leaf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.getcwd()+'/Data/Training_labels_parents'+'.pickle', 'wb') as handle:  # save rgb to label dic
        pickle.dump(labels_parent, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return Features_leaf, Features_parent, labels_leaf, labels_parent



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
    
    leaf_likelihood = {}
    for label in np.unique(labels_leaf):
        bin_labels = binarize_label(labels_leaf,label)
        
        trees = boosted_tree(Features_leaf,bin_labels,nb_weak_learners)
        leaf_likelihood[label] = trees
        
    return leaf_likelihood

### Training

path = os.getcwd()
db_path = path + '/MSRC_ObjCategImageDatabase_v2'  

#create_train_test_split(db_path) # Deja fait
with open(os.getcwd()+'/Data/train_names'+'.pickle', 'rb') as handle:  # save rgb to label dic
    training_names = pickle.load(handle)
with open(os.getcwd()+'/color_to_label'+'.pickle', 'rb') as handle:  # save rgb to label dic
    color_to_label = pickle.load(handle)

N_labels = len(color_to_label)
Features_leaf, Features_parent, labels_leaf, labels_parent = create_matrices(db_path,training_names,N_labels)

parents_likelihood = train_parents_likelihood_dist(Features_parent,labels_parent,nb_weak_learners=5)
with open(os.getcwd()+'/Model/parents_likelihood_debug'+'.pickle', 'wb') as handle:  # save rgb to label dic
    pickle.dump(parents_likelihood, handle, protocol=pickle.HIGHEST_PROTOCOL)

leaf_likelihood = train_leaf_likelihood_dist(Features_leaf,labels_leaf,nb_weak_learners=5)
with open(os.getcwd()+'/Model/leaf_likelihood_debug'+'.pickle', 'wb') as handle:  # save rgb to label dic
    pickle.dump(leaf_likelihood, handle, protocol=pickle.HIGHEST_PROTOCOL)
