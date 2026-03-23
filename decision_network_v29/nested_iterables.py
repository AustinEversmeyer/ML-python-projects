import numpy as np

def flatten(x):
  """
  A function for flattening a nested list.
    
  Inputs:
  -------
  x         : list of list of list of variable
  
  Outputs:
  --------
  y         : list of variable, flattened list
  
  Example:
  --------
  x = [1,[2,3],[4,[5,6]],7]
  returns y = [1,2,3,4,5,6,7]
  """      
  y = []
  for w in x:
    if isinstance(w,list) or isinstance(w,np.ndarray):
      y.extend(flatten(w))
    else:
      y.append(w)
  return y
  
def flatten_enumerate(x,root=True):
  """
  A function for flattening a nested list and returning the indices.
    
  Inputs:
  -------
  x         : list of list of list of variable
  
  Outputs:
  --------
  y         : list of variable, flattened list
  
  Example:
  --------
  x = [1,[2,3],[4,[5,6]],7]
  returns zip(iy,y) where
          y = [1,2,3,4,5,6,7] and
         iy = [[0],[1,0],[1,1],[2,0],[2,1,0],[2,1,1],[3]]
  """          
  iy,y = [],[]
  for iw,w in enumerate(x):
    if isinstance(w,list) or isinstance(w,np.ndarray):
      iu,u = flatten_enumerate(w,root=False)
      y.extend(u)
      iy.extend([iw] + iv for iv in iu)
    else:
      iy.append([iw])
      y.append(w)
  if root:
    return zip(iy,y)
  else:
    return iy,y
    
def get_nested_item(x,iis):
  """
  A function for accessing an item in a nested list
    
  Inputs:
  -------
  x         : list of list of list of variable
  iis       : list of indices
  
  Outputs:
  --------
  _         : variable
  
  Example:
  --------
  x = [1,[2,3],[4,[5,6]],7]
  iis = [2,1,1]
  returns 6
  """              
  if len(iis) == 1:
    return x[iis[0]]
  else:
    return get_nested_item(x[iis[0]],iis[1:])