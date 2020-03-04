import numpy as np

def train_sdm(train_img, train_init_shape, train_gt_shape, cr_model):
  num_train_samples = len(train_img)
  current_shape=np.zeros(train_init_shape.shape)
  cr_model['Step']=[]
  for i in range(num_train_samples):
    current_shape[:,i]=train_init_shape[:,i]
  error_shape = np.zeros(current_shape.shape)
  for step in range(cr_model['deepth']):
    feature_matrix = []
    for i in range(num_train_samples):
      error_shape[:,i] = train_gt_shape[:, i] - current_shape[:,i]

      feature_matrix.append(obtain_HOG(train_img[i], current_shape[:,i], cr_model))

    feature_matrix=np.array(feature_matrix).T
    feature_matrix=np.vstack((feature_matrix,np.ones(feature_matrix.shape[1])))

    penalty=np.eye(feature_matrix.shape[0])*cr_model['lambda']
    cr_model['Step'].append(np.linalg.inv(feature_matrix @ feature_matrix.T+penalty)@ feature_matrix@error_shape.T)
    Bk=np.zeros(current_shape.shape)
    for i in range(Bk.shape[1]):
      Bk[:,i]=cr_model['Step'][step][-1,:]
    current_shape = current_shape +  np.array(cr_model['Step'][step][:-1,:]).T @ feature_matrix[:-1,:] + Bk
  return cr_model
