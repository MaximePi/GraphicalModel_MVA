import os
import pickle
import numpy as np
import cv2
import train

def render_prediction(graph, color_to_label):
    
    label_to_color = {l:c for (c,l) in color_to_label.items()}
    shape = graph.leaf_vertices[0].im_shape
    image = np.zeros((shape[0],shape[1],3))
    
    for leaf in graph.leaf_vertices:
        for p in leaf.mask:
            i,j = p
            color = label_to_color[leaf.get_label()].split('[')[-1]
            color = color.split(']')[0]
            color = color.split(" ")
            color = [c for c in color if len(c) > 0]
            image[i,j,:] = [int(color[0]), int(color[1]), int(color[2])]
    
    return image

def N_m_n(context_matrix,ground_truth,m,n):

  index = len(np.intersection(np.argwhere( context_matrix == n ) , np.argwhere( ground_truth == m )))
  return index

def GPA(context_matrix,ground_truth,labels):
  
  num = np.sum([N_m_n(context_matrix,ground_truth,m,m) for m in labels])
  den = np.sum([N_m_n(context_matrix,ground_truth,m,n) for m in labels for n in labels])
  return num/den

def ACA(context_matrix,ground_truth,labels):

  nb_labels = len(labels)
  num = (1/nb_labels)*np.sum([N_m_n(context_matrix,ground_truth,m,m) for m in labels])
  aca = []
  for m in labels :
    den = np.sum([N_m_n(context_matrix,ground_truth,m,n) for n in labels]) 
    aca.append(num/den)
  return np.mean(aca)

def mIoU(context_matrix,ground_truth,labels):
  nb_labels = len(labels)
  num = (1/nb_labels)*np.sum([N_m_n(context_matrix,ground_truth,m,m) for m in labels])
  mIoU = []
  for m in labels :
    den = np.sum([N_m_n(context_matrix,ground_truth,m,n) + N_m_n(context_matrix,ground_truth,n,m)  for n in labels]) - N_m_n(context_matrix,ground_truth,m,m)
    mIoU.append(num/den)
  return np.mean(mIoU)

### Testing

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

def test(db_path,testing_name,N_labels, color_to_label,leaf_likelihood,parents_likelihood):

    metrics_tot = []
    #images_name = os.listdir(db_path+'/Images')
    #images_name = [image_name for image_name in images_name if image_name in testing_name]
    images_name = ['2_28_s.bmp'] # remove to process all images
    
    for image_name in images_name:
        print(image_name)
        graph_path = db_path+'/FSG_graphs_final'
        
        with open(graph_path+'/'+image_name.split('.')[0]+'.pickle', 'rb') as handle: # load graph with labels
            G = pickle.load(handle)

        newG = train.infer_labels(G, color_to_label, leaf_likelihood,parents_likelihood)
        image = render_prediction(newG, color_to_label)
        
        ground_truth = cv2.imread(db_path+'/GroundTruth/'+image_name.split('.')[0]+'_GT.bmp')
        
        with open(os.getcwd()+'/Data/' + image_name +'.pickle', 'wb') as handle:
            pickle.dump(newG, handle, protocol=pickle.HIGHEST_PROTOCOL)
        cv2.imwrite(os.getcwd()+'/Data/' + image_name +'.bmp', image)
        
        A = np.array([[image[i,j,:] for i in range(image.shape[0])] for j in range(image.shape[1])])
        labels = list(np.unique(A))
        
        metrics = [GPA(image,ground_truth,labels),
                  ACA(image,ground_truth,labels),
                   mIoU(image,ground_truth,labels)]
        
        with open(os.getcwd()+'/Data/' + image_name + "_metrics" +'.pickle', 'wb') as handle:  # save rgb to label dic
            pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        metrics_tot.append(metrics)
    
    return metrics_tot
    
test(db_path,testing_names,N_labels, color_to_label,leaf_likelihood,parents_likelihood)