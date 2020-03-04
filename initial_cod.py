import numpy as np
def initial_cod(In_Shape, Bbox):
  len_points=int(In_Shape.shape[0]/2)
  Out_Shape = np.zeros((len_points,2))
  Out_Shape[:,0] = In_Shape[:len_points]
  Out_Shape[:,1] = In_Shape[len_points:]
  Out_Shape[:,0] = Out_Shape[:,0] / (max(Out_Shape[:,0]) - min(Out_Shape[:,0])) * Bbox[2]
  Out_Shape[:,1] = Out_Shape[:,1] / (max(Out_Shape[:,1]) - min(Out_Shape[:,1])) * Bbox[3]
  Out_Shape[:,0] = Out_Shape[:,0] - min(Out_Shape[:,0]) + Bbox[0]
  Out_Shape[:,1] = Out_Shape[:,1] - min(Out_Shape[:,1]) + Bbox[1]
  return Out_Shape.flatten()
