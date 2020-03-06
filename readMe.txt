TRANSFORMATIONS.PY

Effectue des MeanShift transformations en cascade sur les images situées dans MSRC_ObjCategImageDatabase_v2/Images
Save les images segmentées et les labels/niveau de decomp (format pickle) pour chaque level de transformations dans MSRC_ObjCategImageDatabase_v2/Decomposed_Images

Meta argument à appeler avec transformation.py 

--level : Number de décompositions
--sr_inc : spherical radius increase
--rr_inc : radial radius increase
--den_inc : minimum density increase
