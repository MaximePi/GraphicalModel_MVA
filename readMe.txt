Course Project : Discrete Inference and Learning K.Alahari, G.Charpiat

Based on the work of :
Multi-scale context for scene labeling via ﬂexible segmentation graph Quan Zhoua,n, Baoyu Zhenga, Weiping Zhub, Longin Jan Lateckic 

Packages used :
os, numpy, cv2, pymeanshift, sklearn, scipy, skimage

*All the present scripts were written by ourselves except for pymeanshift.py

Preprocessing.py : segmentation/graph creation/ feature extraction
training_routine.py : train_test split/matrices creation/ classifier training
testing_routine.py : context inference/ image reconstruction/ metrics computation

Results stored in Data/
Initial Database + Segmentations + raw FSGs + FSGs w. features stored in MSRC_ObjCategImageDatabase_v2/
Trained models stored in Model/
*Some data in those folders may not be up to date with the lastest version of the report given the very voluminous FSG with features (40mb each) and the training matrices.