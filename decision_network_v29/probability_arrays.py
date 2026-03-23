import numpy as np
import itertools

def get_dimensions(table):
  """
  A function for getting the dimensions of a nested table or iterable.  No zero
  length dimensions.  Does not check for non uniform dimensions.
    
  Inputs:
  -------
  table         : numpy multidimensional array or iterable
  
  Outputs:
  --------
  dims          : list of integer, dimensions of the table
  """
  
  if hasattr(table,'__iter__'):
    dims = [len(table)] + get_dimensions(table[0])
  else:
    dims = []
  return dims

def expand_ptable(table):
  """
  A function for adding an additional value to the last dimension of a
  probability table and summing probabilities to 1.
    
  Inputs:
  -------
  x         : numpy multidimensional array or iterable, probability table
  
  Outputs:
  --------
  y         : numpy multidimensional array, probability table with additional
              values in the last dimension
  """
  
  dims = get_dimensions(table)
  new_dims = [x + int(i==(len(dims)-1)) for i,x in enumerate(dims)]
  y = np.zeros(new_dims)
  for w in itertools.product(*[range(d) for d in dims[:-1]]):
    for z in range(dims[-1]):
      if w:
        y[*w,z] = table[*w,z]
      else:
        y[z] = table[z]
    if w:
      y[*w,dims[-1]] = 1.0 - sum(y[*w])
    else:
      y[dims[-1]] = 1.0 - sum(y)
  return y

def normalize_cpt(x):
  """
  A function for normalizing a conditional probability table so that the sum
  of entries in the last column is 1.
    
  Inputs:
  -------
  x         : numpy multidimensional array, conditional probability table
  
  Outputs:
  --------
  y         : numpy multidimensional array, normalized
  """      
  dims = x.shape
  y = np.zeros(dims)
  for iis in itertools.product(*(list(range(d)) for d in dims[:-1])):
    s = sum(x[*iis])
    if s > 0:
      for jj in range(dims[-1]):
        y[*iis,jj] = x[*iis,jj]/s
    else:
      y[*iis,0] = 1
  return y
  
def random_cpt(dims):
  """
  A function for generating a random conditional probability table of the
  specified dimensions.
    
  Inputs:
  -------
  x         : iterable of integer, dimensions for cpt
  
  Outputs:
  --------
  y         : numpy multidimensional array, cpt
  """      
  z = np.random.rand(*dims)
  return normalize_cpt(z)
  
def collapse_ptable(z,dim_flags,normalize=False):
  """
  A function for collapsing a table by summing over the other dimensions.
    
  Inputs:
  -------
  z         : numpy multidimensional array, probability table
  dim_flags : iterable of logical, True = keep dimension, False = collapse
  normalize : logical, normalize the last column of the output table
  
  Outputs:
  --------
  w         : numpy multidimensional array, collapsed probability table
  """      
  all_dims = z.shape
  cll_dims = [x for x,y in zip(all_dims,dim_flags) if y]
  w = np.zeros(cll_dims)
  for iis in itertools.product(*(list(range(d)) for d in all_dims)):
    c = [x for x,y in zip(iis,dim_flags) if y]
    w[*c] = w[*c] + z[*iis]
  if normalize:
    w = normalize_cpt(w)
  return w
  
def geometric_fineleft(a,b,n,x):
  """
  A function for providing samples from a to b with a geometric ratio with 
  fine sampling at the left end-point a.
  
  Inputs:
  -------
  a       : float, left end-point
  b       : float, right end-point
  n       : integer, number of terms
  x       : float, geometric ratio, > 1
  
  Outputs:
  --------
  y       : list of float, length n 
  """
  
  z = x**(n-1)
  y = [a + (b-a)*(x**p-1)/(z-1) for p in range(n)]
  return y
  
def geometric_fineright(a,b,n,x):
  
  y = geometric_fineleft(b,a,n,x)
  z = y[::-1]
  return z
  
def geometric_fineboth(a,b,n,x):
  
  if n%2 == 1:
    y = geometric_fineleft(a,(a+b)/2,int((n+1)/2),x)
    z = geometric_fineright((a+b)/2,b,int((n+1)/2),x)
    return y[:-1] + z
  else:
    g = (x**(n/2) - x**((n/2-1)))*(b-a)/(x**(n/2)-1)
    y = geometric_fineleft(a,(a+b-g)/2,int(n/2),x)
    z = geometric_fineright((a+b+g)/2,b,int(n/2),x)
    return y + z