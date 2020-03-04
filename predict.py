import numpy as np
import cv2
import fit_sdm as fit_sdm
import initial_cod as initial_cod
def predict(input_img,cr_model):
  img = input_img.copy()
  face_cascade = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
  face_rects = face_cascade.detectMultiScale(img)[0]

  init_shape = initial_cod.initial_cod(cr_model['mean_shape'], face_rects)
  predict=fit_sdm.fit_sdm(img[:,:,0], init_shape, cr_model)
  x=np.floor(predict[:29]).astype(np.int32)
  y=np.floor(predict[29:]).astype(np.int32)
  for i in range(len(x)):
    cv2.circle(img,(x[i],y[i]),5,(0,0,255),-1)

  cv2.rectangle(img, (face_rects[0],face_rects[1]), (face_rects[0]+face_rects[2],face_rects[1]+face_rects[3]), (255,255,255), 5)

  return img
