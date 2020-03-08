import pickle
from PIL import Image
import cv2
import numpy as np
import os

class SuperPixel:
    def __init__(self,mask,im_shape:tuple):
        self.mask = mask ## coordinates ofthe pixels belonging to the superpixel, shape (N_pixel,2)
        self.im_shape = im_shape # shape of size of the original image

class FSG:
    def __init__(self):
        
        self.leaf_vertices = []
        self.parent_vertices = []
        
    def add_leaf(self,leaf:SuperPixel):
        self.leaf_vertices.append(leaf)

    def add_parent(self,parent:SuperPixel):
        self.parent_vertices.append(parent)

    def instanciate_edges(self):
        self.edges = np.zeros((len(self.leaf_vertices),len(self.parent_vertices)))
        self.leaf2id = { leaf:i for i,leaf in enumerate(self.leaf_vertices)} 
        self.parent2id = { parent:i for i,parent in enumerate(self.parent_vertices)}
        
    def add_edge(self,leaf:SuperPixel,parent:SuperPixel):
        self.edges[self.leaf2id[leaf],self.parent2id[parent]] = 1
        
    
def create_Graph(transformations:dict):

    ## transformations : dictionnaire
    ## clé : niveau de la transformation, associée à : [labels,nb_labels]

    # Add vertices
    Graph = FSG()
    for level,transf in transformations.items():
        labels,nb_labels = transf

        if level==1:  ## coarser transformations, define the leaf level
            for i in range(nb_labels):
                mask = np.argwhere(labels==i)
                Graph.add_leaf(SuperPixel(mask,labels.shape))

        else: ## other transformations, belong to parent vertices
            for i in range(nb_labels):
                mask = np.argwhere(labels==i)
                Graph.add_parent(SuperPixel(mask,labels.shape))

    # Add edges
    Graph.instanciate_edges()
    for leaf in Graph.leaf_vertices:
        for parent in Graph.parent_vertices:
            
            mask_l = np.copy(leaf.mask)
            mask_p = np.copy(parent.mask)
            isIn = np.zeros(mask_l.shape[0])
            
            for i,coordinates_l in enumerate(mask_l):
                ind = np.argwhere(np.all(mask_p == coordinates_l,axis=1)==True)
                if len(ind)!=0:
                    isIn[i] = 1
                else:
                    break
            if np.all(isIn):
                Graph.add_edge(leaf,parent)
                
            
    return Graph




### test
path = os.getcwd()
MSRC_path = path + '/MSRC_ObjCategImageDatabase_v2'
if not os.path.isdir(MSRC_path+'/Decomposed_Images'):
     os.mkdir(MSRC_path+'/Decomposed_Images')
images_name = os.listdir(MSRC_path+'/Images')
image_name = images_name[10]
im_path = MSRC_path+'/Decomposed_Images'+'/'+image_name.split('.')[0]
with open(im_path+'/'+image_name.split('.')[0]+'.pickle', 'rb') as handle:
    transformations = pickle.load(handle)

#transformations = {1 : transformations[1],4 : transformations[4]}


G = create_Graph(transformations)
print(np.sum(G.edges,axis=1))

#print(len(G.parent_vertices))

#A = np.array([[10,55],[84,98],[42,88]])
#B = np.array([[55,10],[42,88],[31,30]])

#print(len(np.argwhere(np.all(B == [88,88],axis=1)==True)))

#print(np.all(C,axis=1))
