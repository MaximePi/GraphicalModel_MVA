import numpy as np
from skimage.color import rgb2yiq

def create_graph_for_graph_cut(rgb, k=1., sigma=0.8, sz=1):
    # create the pixel graph with edge weights as dissimilarities
     
     yuv = rgb2yiq(rgb)

     edges = []
     vertices = []

     for i in range(yuv.shape[0]):
         for j in range(yuv.shape[1]):
             #compute edge weight for nbd pixel nodes for the node i,j
             for i1 in range(i-1, i+2):
                 for j1 in range(j-1, j+2):

                     if i1 == i and j1 == j:
                        continue
                     
                     if 0 <= i1 and i1 < yuv.shape[0] and 0 <= j1 and j1 < yuv.shape[1]:
                        wt = np.abs(yuv[i,j,0]-yuv[i1,j1,0])
                        vertices.append((i,j))
                        vertices.append((i1,j1))
                        edges.append((wt,(i1,j1),(i,j)))
     
     return edges, list(set(vertices))

def MInt(W1, W2, k):
      if len(W1) != 0:
          I_1 = max(W1)
          tau1 = k / len(W1)
          s1 = I_1 + tau1
      else:
          s1 = 1e10
      
      if len(W2) != 0:
          I_2 = max(W2)
          tau2 = k / len(W2)
          s2 = I_2 + tau2
      else:
          s2 = 1e10

      return max(s1, s2)

def refine_graph_for_graph_cut(edges, vertices, k):
    #step 0
    edges.sort()
    #step 1
    Sq = {i:[vertices[i]] for i in range(len(vertices))}
    Wq = {i:[] for i in range(len(vertices))}
    V2Cq = {vertices[i]:i for i in range(len(vertices))}
    #step 2
    for q in range(len(edges)):
        #step 3
        w, v_i, v_j = edges[q]
        c_i, c_j = V2Cq[v_i], V2Cq[v_j]

        if c_i != c_j and w < MInt(Wq[c_i], Wq[c_j], k):
            for v in Sq[max(c_i, c_j)]:
              V2Cq[v] = min(c_i, c_j)
            Sq[min(c_i, c_j)] = Sq[c_i] + Sq[c_j]
            Wq[min(c_i, c_j)] = Wq[c_i] + Wq[c_j] + [w]
            del Sq[max(c_i, c_j)]
            del Wq[max(c_i, c_j)]

    #step 4
    return Sq

# k = 0.005
# image =  cv2.imread("test2.jfif")
# image = cv2.GaussianBlur(image, (5,5),1)
# edges, vertices = create_graph_for_graph_cut(image, k=1., sigma=0.8, sz=1)
# colors = refine_graph_for_graph_cut(edges, vertices, k)
# color_graph(colors, image.shape[:2])
    
# path = os.getcwd()
# db_path = path + '\\MSRC_ObjCategImageDatabase_v2'           
# create_graph(db_path)  