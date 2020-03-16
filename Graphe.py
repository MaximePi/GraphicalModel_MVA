import pickle
import cv2
import numpy as np
import os
import scipy.stats
import scipy.ndimage
import skimage.feature
from scipy import signal
from skimage.color import rgb2yiq
import matplotlib.pyplot as plt
from math import exp, sqrt, pi

class Features:
    def __init__(self,feature1low,feature1high,class1,feature2low,feature2high,class2):
        self.feature1low =feature1low
        self.feature1high=feature1high
        self.class1=class1
        self.feature2low=feature2low
        self.feature2high=feature2high
        self.class2=class2

class SuperPixel:
    def __init__(self,mask,im_shape:tuple,features):
        self.mask = mask ## coordinates ofthe pixels belonging to the superpixel, shape (N_pixel,2)
        self.im_shape = im_shape # shape of size of the original image
        self.feature = features # list of floats 
        self.id = 0
    
    def get_features(self, max_nb_class):
        result = self.feature.feature1low + self.feature.feature1high
        if self.feature.class1 is None:
            return result
        else:
            self.feature.class1 = list(np.zeros((max_nb_class)))
            self.feature.class1[self.label] = 1
            self.feature.class2 = self.feature.class2[:max_nb_class] 
            return result + self.feature.class1 + self.feature.feature2low + self.feature.feature2high + self.feature.class2 

    def set_id(self, id):
        self.id = id
    
    def get_id(self):
      return self.id

    def add_feature(self, features):
        self.feature = features
    
    def set_nb_labels(self, nb_labels):
        self.nb_labels = nb_labels
    
    def get_nb_labels(self):
        return self.nb_labels
    
    def add_label(self,label:int):
        self.label = label
    
    def set_label(self,label:int):
        self.label_pred = label
    
    def set_label_pred(self):
        return self.label_pred
        
    def get_label(self):
        return self.label
    

class FSG:
    def __init__(self, original_image):
        self.leaf_vertices = []
        self.parent_vertices = []
        self.original_image = original_image
        self.leaves_neighbours = {}
        self.parents_neighbours = {}
        
    def add_leaf(self,leaf:SuperPixel):
        self.leaf_vertices.append(leaf)
    
    def add_leaves_neigh(self,neigh):
        self.leaves_neighbours.update(neigh)

    def add_parents_neigh(self,neigh):
        self.parents_neighbours.update(neigh)

    def add_parent(self,parent:SuperPixel):
        self.parent_vertices.append(parent)

    def get_leaf_by_id(self, i):
        return self.leaf_vertices[i]
    
    def get_parent_by_id(self, i):
        return self.leaf_vertices[i]

    def get_edges(self):
        return self.edges

    def get_nb_leaves(self):
        return len(self.leaf_vertices)

    def instanciate_edges(self):
        self.edges = np.zeros((len(self.leaf_vertices),len(self.parent_vertices)))
        for i,leaf in enumerate(self.leaf_vertices):
            leaf.set_id(i)
        for i,parent in enumerate(self.parent_vertices):
            parent.set_id(i) 
        
    def add_edge(self,leaf:SuperPixel,parent:SuperPixel):
        self.edges[leaf.get_id(),parent.get_id()] = 1
    
    def delete_edge(self,leaf:SuperPixel,parent:SuperPixel):
        self.edges[leaf.get_id(),parent.get_id()] = 0
        
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


def create_initial_graph(original_image, transformations:dict):

    ## transformations : dictionnaire
    ## clé : niveau de la transformation, associée à : [labels,nb_labels]

    # Add vertices
    Graph = FSG(original_image)
    k = 0
    for level,transf in transformations.items():
        labels,nb_labels, neigh = transf[0], transf[1], transf[2]
        
        if level==1:  ## coarser transformations, define the leaf level
            for i in range(nb_labels):
                mask = np.argwhere(labels==i)
                leaf = SuperPixel(mask,labels.shape,Features(None,None,None,None,None,None))
                leaf.set_nb_labels(nb_labels)
                Graph.add_leaf(leaf)
                Graph.add_leaves_neigh(neigh)
                
                
        else: ## other transformations, belong to parent vertices
            for i in range(nb_labels):
                mask = np.argwhere(labels==i)
                parent = SuperPixel(mask,labels.shape,Features(None,None,None,None,None,None))
                parent.set_nb_labels(nb_labels)
                Graph.add_parent(parent)
                Graph.add_parents_neigh({(i+k):h for (i,h) in neigh.items()})
                k += len(neigh.keys())

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

def bof(image):
    im = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(im, (14 * 4, 28*4))
    descs =  skimage.feature.daisy(gray, radius = 2, step = 4, rings = 1, histograms=1, orientations=1)
    return [descs[i][j][k] for i in range(len(descs)) for j in range(len(descs[0])) for k in range(len(descs[0][0]))]

def low_features(image, total_size, x_upper_left, y_upper_left):
    
    return hsv_components(image) + CIELab_components(image) + size_ratio(image, total_size) + rgb_components(image) + textures(image) + position(total_size,x_upper_left, y_upper_left) + bof(image)

def add_features_to_parent(graph, parent, neigh):
    #low
    if parent.feature.feature1low is None:
        image, x_upper_left, y_upper_left = graph.convert_superpixel_to_image(parent)
        parent.feature.feature1low =  low_features(np.uint8(image), parent.im_shape, x_upper_left, y_upper_left)
        
    #high
    feature1high = np.array([0. for i in range(858)])
    for id_neigh in neigh:
        n = graph.get_parent_by_id(id_neigh)
        if n.feature.feature1low is None:
            image, x_upper_left, y_upper_left = graph.convert_superpixel_to_image(n)
            n.feature.feature1low =  low_features(np.uint8(image), n.im_shape, x_upper_left, y_upper_left)
        
        feature1high += np.array(n.feature.feature1low)
    
    parent.feature.feature1high = list(feature1high/len(neigh))


def create_segmentation_graph(graph,graph_path):
    
    ## features on leaf vertices shaped (num_classes,2*num_low_features) one line per label of the parents
    ## features on parent vertices shape (1,num_low_features)

    ## adding low level features on parents
    for parent in graph.parent_vertices:
        add_features_to_parent(graph, parent, graph.parents_neighbours[parent.get_id()])
             
    ## adding low+high level features on leaves
    for leaf in graph.leaf_vertices:
        add_features_to_leaf(graph, leaf, graph.leaves_neighbours[leaf.get_id()])

    with open(graph_path, 'wb') as handle: # save Graph
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)  


def add_features_to_leaf(graph, leaf, neigh):
    #low
    if leaf.feature.feature1low is None:
        image, x_upper_left, y_upper_left = graph.convert_superpixel_to_image(leaf)
        leaf.feature.feature1low =  low_features(np.uint8(image), leaf.im_shape, x_upper_left, y_upper_left)
        
    feature1low = leaf.feature.feature1low
    
    class1 = [0 for i in range(leaf.get_nb_labels())]
    class1[leaf.get_label()] = 1
    
    #high
    feature2low = np.array([0. for i in range(858)])
    for id_neigh in neigh:
        n = graph.get_leaf_by_id(id_neigh)
        if n.feature.feature2low is None:
            image, x_upper_left, y_upper_left = graph.convert_superpixel_to_image(n)
            n.feature.feature1low =  low_features(np.uint8(image), n.im_shape, x_upper_left, y_upper_left)
        
        feature2low += np.array(n.feature.feature1low)
    
    feature2low /= len(neigh)
    
    feature2high = np.array([0. for i in range(858)])
    for id_neigh in neigh:
        n = graph.get_parent_by_id(id_neigh)
        if n.feature.feature1high is None:
            image, x_upper_left, y_upper_left = graph.convert_superpixel_to_image(n)
            n.feature.feature1high = low_features(np.uint8(image), n.im_shape, x_upper_left, y_upper_left)
        
        feature2high +=np.array( n.feature.feature1high)
    
    feature2high /= len(neigh)
    
    class2 = np.array([0. for i in range(leaf.get_nb_labels())])
    for id_neigh in neigh:
        n = graph.get_parent_by_id(id_neigh)
        if n.feature.class2 is None:
            n.feature.class2  = [0 for i in range(n.get_nb_labels())]
            n.feature.class2[n.get_label()] = 1
        
        class2 += np.array(n.feature.class2)
    
    parents = [parent for parent in graph.parent_vertices if graph.get_edges()[leaf.id,parent.id] == 1]
    feature1high = np.array([0. for i in range(858)])
    first = True
    for i,parent_node in enumerate(parents):
        if parent_node.feature.feature1low is None:
            image, x_upper_left, y_upper_left = graph.convert_superpixel_to_image(parent_node)
            n.feature.feature1low =  low_features(np.uint8(image), parent_node.im_shape, x_upper_left, y_upper_left)
        
        feature1high = n.feature.feature1low
        if first:
            first = False
            leaf.add_feature(Features(list(feature1low),list(feature1high),list(class1),list(feature2low),list(feature2high),list(class2)))
        else:
            new_leaf = SuperPixel(leaf.mask, leaf.im_shape, Features(list(feature1low),list(feature1high),list(class1),list(feature2low),list(feature2high),list(class2)))
            new_leaf.set_id(graph.get_nb_leaves() + 1)
            new_leaf.add_label(leaf.get_label())
            new_leaf.set_nb_labels(leaf.get_nb_labels())
            #graph.add_leaf(new_leaf)
            #graph.leaves_neighbours[new_leaf.get_id()] = []
            #graph.edges = np.vstack([graph.edges, graph.edges[leaf.get_id(),:].copy()])
            #graph.add_edge(new_leaf,parent_node)
            #graph.delete_edge(leaf,parent_node)


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
        parent.set_nb_labels(len(labels.values()))
        
    return color_to_label

def create_graph(db_path,color_to_label={}, image_names):
    
    ## db_path : Database path
    ## color_to_label : dictionnary linking rgb triplet to label number
    ##      by default is built as the images are processed, save 
    
    if not os.path.isdir(db_path+'/FSG_graphs'):
        os.mkdir(db_path+'/FSG_graphs')
    #images_name = os.listdir(db_path+'/Images')[100:]
    #images_name = [im for im in images_name if im!='Thumbs.db']
    #images_name = ['8_21_s.bmp']
    #images_name = ['2_28_s.bmp','4_8_s.bmp', '6_15_s.bmp'] # remove to process all images
    

    for image_name in images_name:
        print(db_path+'/Decomposed_Images'+'/'+image_name.split('.')[0])
        transformation_path = db_path+'/Decomposed_Images'+'/'+image_name.split('.')[0]
        graph_path = db_path+'/FSG_graphs'+'/'+image_name.split('.')[0]

#        if os.path.isdir(graph_path): # delete previous transfor
#            shutil.rmtree(graph_path)
#        os.mkdir(graph_path)
    
        original_image = cv2.imread(db_path+'/Images/'+image_name)  # load image
        ground_truth = cv2.imread(db_path+'/GroundTruth/'+image_name.split('.')[0]+'_GT.bmp')
        
        with open(transformation_path+'/'+image_name.split('.')[0]+'.pickle', 'rb') as handle: # load decomposed image
            transformations = pickle.load(handle)
        
        G = create_initial_graph( original_image, transformations) # create graph
        color_to_label = labeling(G,ground_truth,color_to_label) # add labels from ground_truth

        with open(graph_path+'.pickle', 'wb') as handle: # save Graph
            pickle.dump(G, handle, protocol=pickle.HIGHEST_PROTOCOL)  
    with open(os.getcwd()+'/color_to_label'+'.pickle', 'wb') as handle:  # save rgb to label dic
        pickle.dump(color_to_label, handle, protocol=pickle.HIGHEST_PROTOCOL)

def render_prediction(graph, color_to_label):
    
    label_to_color = {l:c for (c,l) in color_to_label.items()}
    image = np.zeros(graph.leaf_vertices[0].im_shape)
    
    for leaf in graph.leaf_vertices:
        for p in leaf.mask:
            i,j = p
            image[i,j] = label_to_color(leaf.get_label())
    
    return image

