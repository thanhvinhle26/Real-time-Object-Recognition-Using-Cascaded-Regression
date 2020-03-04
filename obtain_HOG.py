import numpy as np
from skimage import feature
def obtain_HOG(img,shape,cr_model):
  features = []
  len_points=int(shape.shape[0]/2)
  for i in range(len_points):
    x = np.floor(shape[i])
    y = np.floor(shape[i+len_points])
    IX_X = np.arange(x - cr_model['patch_radius'] + 1,x + cr_model['patch_radius']+1,1)
    IX_Y =np.arange(y - cr_model['patch_radius'] + 1,y + cr_model['patch_radius']+1,1)
    IX_Y[IX_Y<1] = 1
    IX_X[IX_X<1] = 1
    IX_Y[IX_Y>=img.shape[0]] = img.shape[0]-1
    IX_X[IX_X>=img.shape[1]] = img.shape[1]-1
    tmp = np.zeros((len(IX_Y),len(IX_X)))
    for i in range(len(IX_Y)):
      for j in range(len(IX_X)):
        tmp[i,j]=img[int(IX_Y[i]),int(IX_X[j])]
    H = feature.hog(tmp, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2")
    features.append(H)
  hog_feat=[]
  for i in range(len(features)):
    hog_feat.append(features[i].tolist())
  return np.array(hog_feat).flatten()
