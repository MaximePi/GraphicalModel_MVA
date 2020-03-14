import os
import cv2
import pymeanshift as pms
from sklearn.cluster import MeanShift
import numpy as np
from PIL import Image
import pymeanshift
from joblib import dump,load
import pickle
import argparse
import shutil



parser = argparse.ArgumentParser()
parser.add_argument('--level',type = int)
parser.add_argument('--sr_inc',type = int)
parser.add_argument('--rr_inc',type = int)
parser.add_argument('--den_inc',type = int)
opts = parser.parse_args()

##### Decompose les images selon MeanShift
##### Save les labels en pkl

if opts.level!=None:
    decomposition_levels = opts.level
else:
    decomposition_levels = 6
if opts.sr_inc!=None:
    sr_inc = opts.sr_inc
else:
    sr_inc = 2

if opts.rr_inc!=None:
    rr_inc = opts.rr_inc
else:
    rr_inc = 1
if opts.den_inc!=None:
    den_inc = opts.den_inc
else:
    den_inc = 2


path = os.getcwd()
MSRC_path = path + '/MSRC_ObjCategImageDatabase_v2'

if not os.path.isdir(MSRC_path+'/Decomposed_Images'):
     os.mkdir(MSRC_path+'/Decomposed_Images')
images_name = os.listdir(MSRC_path+'/Images')

images_name = [images_name[10],images_name[15],images_name[30]] # remove to process all images


for image_name in images_name:
    im_path = MSRC_path+'/Decomposed_Images'+'/'+image_name.split('.')[0]
    
    if os.path.isdir(im_path): # delete previous transfor
        shutil.rmtree(im_path)
    os.mkdir(im_path)
    
    original_image = cv2.imread(MSRC_path+'/Images/'+image_name)
        
    params = {'spatial_radius' : 1,'range_radius' : 1,'min_density' : 100}
    decompositions = { i+1 : [None,None] for i in range(decomposition_levels)}
    
    for i in range(decomposition_levels):
        
        (segmented_image, labels_image, number_regions) = pms.segment(original_image,**params)
        decompositions[i+1][0] = labels_image
        decompositions[i+1][1] = number_regions
        
        params['spatial_radius'] += sr_inc
        params['range_radius'] += rr_inc
        params['min_density'] += den_inc
        
        im = Image.fromarray(segmented_image)
        im.save(im_path+'/'+image_name.split('.')[0]+'_level_'+str(i)+'.png')

    with open(im_path+'/'+image_name.split('.')[0]+'.pickle', 'wb') as handle:
        pickle.dump(decompositions, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
def transformations(MSRC_path):
    
    den_inc = 50
    rr_inc = 1
    sr_inc = 2
    
    if not os.path.isdir(MSRC_path+'/Decomposed_Images'):
         os.mkdir(MSRC_path+'/Decomposed_Images')
    images_name = os.listdir(MSRC_path+'/Images')
    
    images_name = ['2_29_s.bmp','15_3_s.bmp','18_21_s.bmp'] # remove to process all images
    
    
    for image_name in images_name:
        print(image_name)
        im_path = MSRC_path+'/Decomposed_Images'+'/'+image_name.split('.')[0]
        
        if os.path.isdir(im_path): # delete previous transfor
            shutil.rmtree(im_path)
        os.mkdir(im_path)
        
        original_image = cv2.imread(MSRC_path+'/Images/'+image_name)
            
        params = {'spatial_radius' : 1,'range_radius' : 1,'min_density' : 100}
        decompositions = { i+1 : [None,None] for i in range(decomposition_levels)}
        
        for i in range(decomposition_levels):
            
            (segmented_image, labels_image, number_regions) = pms.segment(original_image,**params)
            decompositions[i+1][0] = labels_image
            decompositions[i+1][1] = number_regions
            
            params['spatial_radius'] += sr_inc
            params['range_radius'] += rr_inc
            params['min_density'] += den_inc
            
            im = Image.fromarray(segmented_image)
            im.save(im_path+'/'+image_name.split('.')[0]+'_level_'+str(i)+'.png')
    
        with open(im_path+'/'+image_name.split('.')[0]+'.pickle', 'wb') as handle:
            pickle.dump(decompositions, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
## A utiliser pour lire les pickles
#with open(im_path+'/'+image_name.split('.')[0]+'.pickle', 'rb') as handle:
#    unserialized_data = pickle.load(handle)

#print(unserialized_data)
