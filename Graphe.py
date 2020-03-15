import pickle
import cv2
import numpy as np
import os
import scipy.stats
import scipy.ndimage
import skimage.feature
from math import sqrt

class SuperPixel:
    def __init__(self,mask,im_shape:tuple,features):
        self.mask = mask ## coordinates ofthe pixels belonging to the superpixel, shape (N_pixel,2)
        self.im_shape = im_shape # shape of size of the original image
        self.feature = features # list of floats 
        self.id = 0

    def set_id(self, id):
        self.id = id
    
    def get_id(self):
      return self.id

    def add_feature(self, features):
        self.feature = features
    
    def get_features(self):
        return self.feature
    
    def add_label(self,label:int):
        self.label = label
    

class FSG:
    def __init__(self, original_image):
        self.leaf_vertices = []
        self.parent_vertices = []
        self.original_image = original_image
        
    def add_leaf(self,leaf:SuperPixel):
        self.leaf_vertices.append(leaf)

    def add_parent(self,parent:SuperPixel):
        self.parent_vertices.append(parent)

    def get_edges(self):
        return self.edges

    def get_nb_leaves(self):
        return len(self.leaf_vertices)

    def get_nb_parents(self):
        return len(self.parent_vertices)

    def get_leaves(self, i):
        return self.leaf_vertices[i]

    def get_parent(self, i):
        return self.parent_vertices[i]

    def change_leaf(self, i, pixel:SuperPixel):
        self.leaf_vertices[i] =pixel

    def instanciate_edges(self):
        self.edges = np.zeros((len(self.leaf_vertices),len(self.parent_vertices)))
        for i,leaf in enumerate(self.leaf_vertices):
            leaf.set_id(i)
        for i,parent in enumerate(self.parent_vertices):
            parent.set_id(i) 
        
    def add_edge(self,leaf:SuperPixel,parent:SuperPixel):
        self.edges[leaf.get_id(),parent.get_id()] = 1
        
    def convert_superpixel_to_image(self, superpixel:SuperPixel):
        m, M = get_pixel(superpixel.mask, [superpixel.mask.shape[0], superpixel.mask.shape[0]], [0, 0])
        image = np.zeros((M[0] - m[0]+1, M[1] - m[1]+1,3))
        
        for p in superpixel.mask:
            i,j = p
            image[i-m[0],j-m[1],:] = self.original_image[i,j,:]
        
        return image, m[0], m[1]

def get_pixel(p,m,M):
    
    for k in range(p.shape[0]):
        i, j = p[k,0], p[k,1]
        if m[0] >= i:
            m[0] = i
        if M[0] <= i:
            M[0] = i
        if m[1] >= j:
            m[1] = j
        if M[1] <= j:
            M[1] = j
    
    return m, M

def create_segmentation_graph(graph,color_2_label,training=True):
    
    ## features on leaf vertices shaped (num_classes,2*num_low_features) one line per label of the parents
    ## features on parent vertices shape (1,num_low_features)
    
    kaze = create_bow()

    for leaf in graph.leaf_vertices: ## adding low+high level features on leaves

        image, x_upper_left, y_upper_left = graph.convert_superpixel_to_image(leaf)
        features = low_features(np.uint8(image), leaf.im_shape, kaze, x_upper_left, y_upper_left)
        features = np.reshape(features*int(sum(graph.edges[leaf.get_id(),:])),(int(sum(graph.edges[leaf.get_id(),:])),len(features)))
        print(int(sum(graph.edges[leaf.get_id(),:])))
        features_high = high_features(graph, image, kaze, leaf,color_2_label)
        features = np.concatenate((features,features_high),axis=1)
        leaf.feature = features
             
    for parent in graph.parent_vertices: ## adding low level features on parents
        image, x_upper_left, y_upper_left = graph.convert_superpixel_to_image(parent)
        features = low_features(np.uint8(image), parent.im_shape, kaze, x_upper_left, y_upper_left)
        parent.add_feature(features)
    
    return graph
    
def create_Graph(original_image, transformations:dict):

    ## transformations : dictionnaire
    ## clé : niveau de la transformation, associée à : [labels,nb_labels]

    # Add vertices
    Graph = FSG(original_image)
    for level,transf in transformations.items():
        labels,nb_labels = transf

        if level==1:  ## coarser transformations, define the leaf level
            for i in range(nb_labels):
                mask = np.argwhere(labels==i)
                Graph.add_leaf(SuperPixel(mask,labels.shape, []))

        else: ## other transformations, belong to parent vertices
            for i in range(nb_labels):
                mask = np.argwhere(labels==i)
                Graph.add_parent(SuperPixel(mask,labels.shape,[]))

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

def apply_routine(in_channels):
    out_channels = []
    for i in range(len(in_channels)):
        out_channels.append(np.mean(in_channels[i]))
        out_channels.append(np.std(in_channels[i]))
        out_channels.append(scipy.stats.skew(in_channels[i],  axis=None))
        out_channels.append(scipy.stats.kurtosis(in_channels[i],  axis=None))
    return out_channels

def CIELab_components(image):
    lab_image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    L,a,b = cv2.split(lab_image)
    output = apply_routine([L,a,b])
    imhist,bins = np.histogram(L, 10)
    return  list(imhist) + list(output)

def hsv_components(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v =  cv2.split(hsv_image)
    output = apply_routine([h,s,v])
    imhist_h,bins = np.histogram(h, 5)
    imhist_s,bins = np.histogram(s, 3)
    return  list(imhist_h) + list(imhist_s) + list(output)

def rgb_components(image):
    p = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    if p.shape[0]<3:
        p = np.concatenate(np.zeros((1,p.shape[1],p.shape[2])),axis=0)
    r,g,b = p[0],p[1],p[2]
    output = apply_routine([r,g,b])
    imhist_r,bins = np.histogram(r, 10)
    imhist_g,bins = np.histogram(g, 10)
    imhist_b,bins = np.histogram(b, 10)
    return list(imhist_r) + list(imhist_g) + list(imhist_b) + list(output)

def size_ratio(image, total_size):
    return [(image.shape[0] * image.shape[1]) / (total_size[0] * total_size[1])]

def position(total_size,x_upper_left, y_upper_left):
    return [x_upper_left, y_upper_left, sqrt((x_upper_left - total_size[0]/2)**2+(y_upper_left - total_size[1]/2)**2)]

def textures(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(image,(5,5),cv2.BORDER_DEFAULT)
    mean =  cv2.blur(image,(5,5))
    median = cv2.medianBlur(image,ksize=3)
    laplace = scipy.ndimage.filters.laplace(image)
    sobel = scipy.ndimage.sobel(image)
    laplace_gaussian = scipy.ndimage.gaussian_laplace(image, 3)
    uniform = scipy.ndimage.uniform_filter(image, size=20)
    median3 = cv2.medianBlur(image,ksize=9)
    gaussian2 = cv2.GaussianBlur(image,(3,3),cv2.BORDER_DEFAULT)
    mean2 =  cv2.blur(image,(3,3))
    median2 = cv2.medianBlur(image,ksize=5)
    prewit = scipy.ndimage.prewitt(image)
    laplace_gaussian2 = scipy.ndimage.gaussian_laplace(image, 5)
    uniform2 = scipy.ndimage.uniform_filter(image, size=10)
    mean3 =  cv2.blur(image,(9,9))
    bilateral = cv2.bilateralFilter(image,3,1,1)
    all_filters = [gaussian, mean, median, laplace, sobel, laplace, laplace_gaussian, uniform, mean3, gaussian2,mean2, median2, prewit, laplace_gaussian2, uniform2, median3, bilateral]
    result = []
    for filt in all_filters:
      image = np.uint8(filt)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      result += apply_routine([gray])
    return result

def create_bow():
    return cv2.AKAZE_create(descriptor_channels = 700)

def bof(image, kaze):
    im = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(im, (14 * 4, 28*4))
    descs =  skimage.feature.daisy(gray, radius = 2, step = 4, rings = 1, histograms=1, orientations=1)
    return [descs[i][j][k] for i in range(len(descs)) for j in range(len(descs[0])) for k in range(len(descs[0][0]))]

def low_features(image, total_size, kaze, x_upper_left, y_upper_left):
    return hsv_components(image) + CIELab_components(image) + size_ratio(image, total_size) + rgb_components(image) + textures(image) + position(total_size,x_upper_left, y_upper_left) + bof(image, kaze)

def high_features(graph, image, kaze, leaf, color_to_label):
    
    ## Output high_features , shape (number of classes, number of features)
    ## One set of averaged features per class of the parent pixels
    
    high_features = np.zeros((sum(graph.edges[leaf.get_id(),:]),858+len(color_to_label)))
    parents = [parent for parent in graph.parent_vertices if edges[leaf.id,parent.id] == 1]
        
    for i,parent_node in enumerate(parents):
        #count_parents += len(parent_node.mask)
        image, x_upper_left, y_upper_left = graph.convert_superpixel_to_image(parent_node)
        feature =  np.array(low_features(np.uint8(image), parent_node.im_shape, kaze, x_upper_left, y_upper_left))
        label_dist = np.zeros(len(color_to_label))
        label_dist[parent_node.label] = 1
        feature = np.concatenate(feature,label_dist)
        high_features[i,:] = feature
        
    return high_features


def labeling(graph:FSG,ground_truth,color_to_label:dict):
    ## Assign a label to the superpixel, dominant label in corresponding groundtruth region
    
    for leaf in graph.leaf_vertices:
        
        labels = {label : 0 for label in range(len(color_to_label))}
        for pos in leaf.mask:
    
            color = ground_truth[pos[0],pos[1],:]
            
            if str(color) in color_to_label.keys():
                labels[color_to_label[str(color)]]+=1
            else:
                color_to_label[str(color)] = len(color_to_label)
                labels[color_to_label[str(color)]]=1
                
        super_pixel_label = np.argmax(list(labels.values()))
        leaf.add_label(super_pixel_label)
        
    for parent in graph.parent_vertices:  
        labels = {label : 0 for label in range(len(color_to_label))}
        for pos in parent.mask:
    
            color = ground_truth[pos[0],pos[1],:]
            
            if str(color) in color_to_label.keys():
                labels[color_to_label[str(color)]]+=1
            else:
                color_to_label[str(color)] = len(color_to_label)
                labels[color_to_label[str(color)]]=1
         
        super_pixel_label = np.argmax(list(labels.values()))
        parent.add_label(super_pixel_label)
        
    return color_to_label

def create_graph(db_path,color_to_label={}):
    
    ## db_path : Database path
    ## color_to_label : dictionnary linking rgb triplet to label number
    ##      by default is built as the images are processed, save 
    
    if not os.path.isdir(db_path+'/FSG_graphs'):
        os.mkdir(db_path+'/FSG_graphs')
    images_name = os.listdir(db_path+'/Images')
    images_name = [im for im in images_name if im!='Thumbs.db']
    #images_name = ['20_9_s.bmp']
    #images_name = ['2_29_s.bmp','15_3_s.bmp','18_21_s.bmp'] # remove to process all images
    

    for image_name in images_name:
        print(image_name)
        transformation_path = db_path+'/Decomposed_Images'+'/'+image_name.split('.')[0]
        graph_path = db_path+'/FSG_graphs'+'/'+image_name.split('.')[0]

#        if os.path.isdir(graph_path): # delete previous transfor
#            shutil.rmtree(graph_path)
#        os.mkdir(graph_path)
    
        original_image = cv2.imread(db_path+'/Images/'+image_name)  # load image
        ground_truth = cv2.imread(db_path+'/GroundTruth/'+image_name.split('.')[0]+'_GT.bmp')
        
        with open(transformation_path+'/'+image_name.split('.')[0]+'.pickle', 'rb') as handle: # load decomposed image
            transformations = pickle.load(handle)
        
        G = create_Graph( original_image, transformations) # create graph
        color_to_label = labeling(G,ground_truth,color_to_label) # add labels from ground_truth

        with open(graph_path+'.pickle', 'wb') as handle: # save Graph
            pickle.dump(G, handle, protocol=pickle.HIGHEST_PROTOCOL)  
    with open(os.getcwd()+'/color_to_label'+'.pickle', 'wb') as handle:  # save rgb to label dic
        pickle.dump(color_to_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
    
#path = os.getcwd()
#db_path = path + '/MSRC_ObjCategImageDatabase_v2'           
#reate_graph(db_path)

## DEbug
            
#path = os.getcwd()
#db_path = path + '/MSRC_ObjCategImageDatabase_v2'           
#image_name = '2_29_s.bmp'
#
#original_image = cv2.imread(db_path+'/Images/'+image_name)  # load image
#ground_truth = cv2.imread(db_path+'/GroundTruth/'+image_name.split('.')[0]+'_GT.bmp')
#transformation_path = db_path+'/Decomposed_Images'+'/'+image_name.split('.')[0]
#
#with open(transformation_path+'/'+image_name.split('.')[0]+'.pickle', 'rb') as handle: # load decomposed image
#    transformations = pickle.load(handle)
#        
#graph = create_Graph(original_image, transformations) # create graph
#color_to_label = {}
#

#create_graph_features(MSRC_path) 


#def 
#path = os.getcwd()
#MSRC_path = path + '/MSRC_ObjCategImageDatabase_v2'
#if not os.path.isdir(MSRC_path+'/Decomposed_Images'):
#     os.mkdir(MSRC_path+'/Decomposed_Images')
#images_name = os.listdir(MSRC_path+'/Images')
#image_name = '15_3_s.bmp'
#im_path = MSRC_path+'/Decomposed_Images'+'/'+image_name.split('.')[0]
#with open(im_path+'/'+image_name.split('.')[0]+'.pickle', 'rb') as handle:
#    transformations = pickle.load(handle)
#
#transformations = {1 : transformations[1],4 : transformations[4]}
#im_path = MSRC_path+'/Images'
#
#original_image =cv2.imread(im_path+'/'+image_name)
#print(original_image)
#G = create_Graph( original_image, transformations)
#print(np.sum(G.edges,axis=1))
#
#G_s = create_segmentation_graph(G)

#print(len(G.parent_vertices))

#A = np.array([[10,55],[84,98],[42,88]])
#B = np.array([[55,10],[42,88],[31,30]])

#print(len(np.argwhere(np.all(B == [88,88],axis=1)==True)))

#print(np.all(C,axis=1))


#    count_parents = 0.00001
#    feature = np.array([0. for i in range(858)])
#    edges = graph.get_edges()
#    parents = [i for i in range(graph.get_nb_parents()) if edges[leaf_id,i] == 1]
#    for parent_id in parents:
#        parent_node = graph.get_parent(parent_id)
#        count_parents += len(parent_node.mask)
#        image, x_upper_left, y_upper_left = graph.convert_superpixel_to_image(parent_node)
#        feature += len(parent_node.mask) * np.array(low_features(np.uint8(image), parent_node.im_shape, kaze, x_upper_left, y_upper_left))
#    return list(feature / count_parents)

    
    
#    for i in range(graph.get_nb_leaves()): ## adding low+high level features on leaves
#        leaf = graph.get_leaves(i)
#        image, x_upper_left, y_upper_left = graph.convert_superpixel_to_image(leaf)
#        features = low_features(np.uint8(image), leaf.im_shape, kaze, x_upper_left, y_upper_left)
#        leaf_id = leaf.get_id()
#        features += high_features(graph, image, kaze, leaf_id)
#        graph.change_leaf(i, SuperPixel(leaf.mask,leaf.im_shape,features))
    
