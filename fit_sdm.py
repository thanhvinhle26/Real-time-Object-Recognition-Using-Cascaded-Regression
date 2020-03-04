import numpy as np
import obtain_HOG as obtain_HOG
def fit_sdm(img,init_shape,cr_model):
  predict_shape = init_shape
  for i in range(cr_model['deepth']):
    feature = obtain_HOG.obtain_HOG(img, predict_shape, cr_model)
    feature=feature.reshape(-1,1)
    predict_shape = predict_shape + (cr_model['Step'][i][:-1,:].T@feature+cr_model['Step'][i][-1,:].reshape(-1,1)).reshape(-1,)

    return predict_shape
