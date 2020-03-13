import pickle
import cv2
import numpy as np
import os
import scipy.stats
import skimage.feature
from math import sqrt
from matplotlib import pyplot as plt

class SuperPixel:
    def __init__(self,mask,im_shape:tuple):
        self.mask = mask ## coordinates ofthe pixels belonging to the superpixel, shape (N_pixel,2)
        self.im_shape = im_shape # shape of size of the original image
    
    def add_feature(self, features):
        self.feature = features
    
    def get_mask(self):
        return self.mask
        
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
    
    def get_leaves(self):
        return self.leaf_vertices

    def instanciate_edges(self):
        self.edges = np.zeros((len(self.leaf_vertices),len(self.parent_vertices)))
        self.leaf2id = { leaf:i for i,leaf in enumerate(self.leaf_vertices)} 
        self.parent2id = { parent:i for i,parent in enumerate(self.parent_vertices)}
        self.id2leaf = {i:parent for i,parent in enumerate(self.parent_vertices)}
        
    def add_edge(self,leaf:SuperPixel,parent:SuperPixel):
        self.edges[self.leaf2id[leaf],self.parent2id[parent]] = 1
        
    def add_features_leaf(self, features, leaf):
        self.features[self.leaf2id[leaf]] = features
        
    def convert_superpixel_to_image(self, superpixel:SuperPixel):
        m, M = get_pixel(superpixel.mask, [superpixel.mask.shape[0], superpixel.mask.shape[0]], [0, 0])
        image = np.zeros((M[0] - m[0]+1, M[1] - m[1]+1,3))
        
        for p in superpixel.mask:
            i,j = p
            image[i-m[0],j-m[1],:] = self.original_image[i,j,:]
        
        return image.astype(np.float32), m[0], m[1]

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

def create_segmentation_graph(graph):

    all_leaves = graph.get_leaves()
    kaze = create_bow()
    
    for edge in all_leaves:
        image, x_upper_left, y_upper_left = graph.convert_superpixel_to_image(edge)
        featuresLow = low_features(image, edge.im_shape, kaze, x_upper_left, y_upper_left)
        edge.add_features_leaf(featuresLow)
    
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


def apply_routine(in_channels):
    out_channels = []
    for i in range(len(in_channels)):
        out_channels.append(np.mean(in_channels[i]))
        out_channels.append(np.std(in_channels[i]))
        out_channels.append(scipy.stats.skew(in_channels[i]))
        out_channels.append(scipy.stats.kurtosis(in_channels[i]))
    return out_channels

def CIELab_components(image):
    L,a,b = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    output = apply_routine([L,a,b])
    imhist,bins = np.histogram(L, 10)
    return  imhist + output

def hsv_components(image):
    h,s,v = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    output = apply_routine([h,s,v])
    imhist_h,bins = np.histogram(h, 5)
    imhist_s,bins = np.histogram(s, 3)
    return  imhist_h + imhist_s + output

def rgb_components(image):
    r,g,b = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    output = apply_routine([r,g,b])
    imhist_r,bins = np.histogram(r, 10)
    imhist_g,bins = np.histogram(g, 10)
    imhist_b,bins = np.histogram(b, 10)
    return imhist_r + imhist_g + imhist_b + output

def size_ratio(image, total_size):
    return [(image.shape[0] * image.shape[1]) / (total_size[0] * total_size[1])]

def position(total_size,x_upper_left, y_upper_left):
    return [x_upper_left, y_upper_left, sqrt((x_upper_left - total_size[0]/2)**2+(y_upper_left - total_size[1]/2)**2)]

def get_texture(image):
    g = skimage.feature.greycomatrix(image, [1, 2], [0, np.pi/2], levels=4, normed=True, symmetric=True)
    return skimage.feature.greycoprops(g, 'contrast')[0]

def textures(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gaussian = get_texture(cv2.GaussianBlur(image,(5,5),cv2.BORDER_DEFAULT))
    mean =  get_texture(cv2.blur(image,(5,5)))
    median = get_texture(cv2.medianBlur(image,ksize=3))
    laplace = get_texture(cv2.Laplacian(gray, cv2.CV_16S, ksize=3))
    sobel = get_texture(scipy.ndimage.sobel(image))
    laplace_gaussian = scipy.ndimage.gaussian_laplace(image, 3)
    harris = get_texture(cv2.cornerHarris(image,4,3,2))
    canny = get_texture(cv2.canny(image, 10,100))
    gaussian2 = get_texture(cv2.GaussianBlur(image,(3,3),cv2.BORDER_DEFAULT))
    mean2 =  get_texture(cv2.blur(image,(3,3)))
    median2 = get_texture(cv2.medianBlur(image,ksize=5))
    laplace2 = get_texture(cv2.Laplacian(gray, cv2.CV_16S, ksize=5))
    laplace_gaussian2 = scipy.ndimage.gaussian_laplace(image, 5)
    harris2 = get_texture(cv2.cornerHarris(image,4,5,3))
    canny2 = get_texture(cv2.canny(image, 100,150))
    bilateral = get_texture(cv2.bilateralFilter(image,3,1,1))
    return [gray, gaussian, mean, median, laplace, sobel, laplace, laplace_gaussian, harris, canny, gaussian2,mean2, median2, laplace2, laplace_gaussian2, harris2, canny2, bilateral]

def create_bow():
    return cv2.AKAZE_create(700)

def bof(image, kaze):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return kaze.detect(gray,None)

def low_features(image, total_size, kaze, x_upper_left, y_upper_left):
    return bof(image, kaze) + hsv_components(image) + CIELab_components(image) + size_ratio(image, total_size) + rgb_components(image) + textures(image) + get_texture(image) + position(total_size,x_upper_left, y_upper_left)

#def high_features(super_pixels):
 #   return sum(intersection)


### test
path = os.getcwd()
MSRC_path = path + '/MSRC_ObjCategImageDatabase_v2'
if not os.path.isdir(MSRC_path+'/Decomposed_Images'):
     os.mkdir(MSRC_path+'/Decomposed_Images')
images_name = os.listdir(MSRC_path+'/Images')
image_name = "15_3_s.jpg"
im_path = MSRC_path+'/Decomposed_Images'+'/'+image_name.split('.')[0]
with open(im_path+'/'+image_name.split('.')[0]+'.pickle', 'rb') as handle:
    transformations = pickle.load(handle)

#transformations = {1 : transformations[1],4 : transformations[4]}
im_path = MSRC_path+'/Images'

original_image = plt.imread((im_path+'\\'+image_name.split('.')[0]+'.bmp').replace('/', "\\"))

G = create_Graph( original_image, transformations)
print(np.sum(G.edges,axis=1))

G_s = create_segmentation_graph(G)

#print(len(G.parent_vertices))

#A = np.array([[10,55],[84,98],[42,88]])
#B = np.array([[55,10],[42,88],[31,30]])

#print(len(np.argwhere(np.all(B == [88,88],axis=1)==True)))

#print(np.all(C,axis=1))
