import unittest
import numpy as np
import itertools
from probability_arrays import normalize_cpt, random_cpt, collapse_ptable, \
                               get_dimensions, geometric_fineleft, \
                               geometric_fineright, geometric_fineboth,\
                               expand_ptable
from nested_iterables import flatten

class TestGetDimensions(unittest.TestCase):
  
  def test_basic(self):
    x = np.array([[2,6],[0,0],[4,3]])
    y = get_dimensions(x)
    expected_y = [3,2]
    self.assertEqual(y,expected_y)
    
    x = [[2,6],[0,0],[4,3]]
    y = get_dimensions(x)
    expected_y = [3,2]
    self.assertEqual(y,expected_y)
    
    x = [np.array([2,6]),np.array([0,0]),np.array([4,3])]
    y = get_dimensions(x)
    expected_y = [3,2]
    self.assertEqual(y,expected_y)
    
class TestExpandPTable(unittest.TestCase):
  
  def test_basic(self):
    x = np.array([0.75])
    y = list(expand_ptable(x))
    expected_y = [0.75,0.25]
    self.assertEqual(len(y),len(expected_y))
    for u,v in zip(y,expected_y):
      self.assertAlmostEqual(u,v,delta=1e-6)
    
    x = np.array([0.75,0.10])
    y = list(expand_ptable(x))
    expected_y = [0.75,0.10,0.15]
    self.assertEqual(len(y),len(expected_y))
    for u,v in zip(y,expected_y):
      self.assertAlmostEqual(u,v,delta=1e-6)
      
    x = np.array([[0.75,0.10],[0.50,0.10]])
    y = list(expand_ptable(x))
    expected_y = [[0.75,0.10,0.15],[0.50,0.10,0.40]]
    self.assertEqual(len(y),len(expected_y))
    for u,v in zip(flatten(y),flatten(expected_y)):
      self.assertAlmostEqual(u,v,delta=1e-6)

class TestNormalizeCPT(unittest.TestCase):
  
  def test_basic(self):
    x = np.array([[2,6],[0,0]])
    y = normalize_cpt(x)
    expected_y = np.array([[0.25,0.75],[1,0]])
    self.assertEqual(y.shape, expected_y.shape)
    self.assertEqual(flatten(y), flatten(expected_y))
    
class TestRandomCPT(unittest.TestCase):
  
  def test_basic(self):
    dims = [4,3,3]
    x = random_cpt(dims)
    
    sums = []
    for iis in itertools.product(*(list(range(d)) for d in x.shape[:-1])):
      sums.append(sum(x[*iis]))
    
    self.assertEqual(x.shape, tuple(dims))
    self.assertEqual(all(1-1e-6 <= x <= 1+1e-6 for x in sums), True)

class TestCollapsePTable(unittest.TestCase):
    
  def test_basic(self):
    table = np.array([[0.75,0.25],[0.5,0.5]])
    flags = [True,False]
    y = collapse_ptable(table,flags)
    expected_y = [1,1]
    self.assertEqual(tuple(y), tuple(expected_y))
    
    table = np.array([[0.75,0.25],[0.5,0.5]])
    flags = [True,False]
    y = collapse_ptable(table,flags,normalize=True)
    expected_y = [0.5,0.5]
    self.assertEqual(tuple(y), tuple(expected_y))
    
    table = np.array([[0.75,0.25],[0.5,0.5]])
    flags = [True,True]
    y = collapse_ptable(table,flags,normalize=True)
    expected_y = np.array([[0.75,0.25],[0.5,0.5]])
    self.assertEqual(y.shape, expected_y.shape)
    self.assertEqual(tuple(flatten(y)), tuple(flatten(expected_y)))
    
    
class GeometricFineLeft(unittest.TestCase):
  
  def test_basic(self):
    a = 0
    b = 1
    x = 10
    n = 11
    y = geometric_fineleft(a,b,n,x)
    expected_y = [0,9e-10,9.9e-9,9.99e-8,9.999e-7,9.9999e-6,9.99999e-5,\
                  9.999999e-4,9.9999999e-3,9.99999999e-2,1]

    self.assertEqual(len(y),len(expected_y))
    for u,v in zip(y,expected_y):
      self.assertAlmostEqual(u,v,delta=0.01*v)
      
class GeometricFineRight(unittest.TestCase):
  
  def test_basic(self):
    a = 0
    b = 1
    x = 10
    n = 11
    y = geometric_fineright(a,b,n,x)
    expected_y = [0,0.9,0.99,0.999,0.9999,0.99999,0.999999,0.9999999,\
                  0.99999999,0.999999999,1]
    self.assertEqual(len(y),len(expected_y))
    for u,v in zip(y,expected_y):
      self.assertAlmostEqual(u,v,delta=0.01*v)
      
class GeometricFineBoth(unittest.TestCase):
  
  def test_basic(self):
    a = 0
    b = 1
    x = 1.5
    n = 21
    y = geometric_fineboth(a,b,n,x)
    expected_y = [0.0, 0.004, 0.01, 0.02, 0.036, 0.058, 0.092, 0.14, 0.22, \
                  0.33, 0.50, 0.67, 0.78, 0.86, 0.91, 0.94, 0.96, 0.98, \
                  0.99, 0.996, 1.0]
    self.assertEqual(len(y),len(expected_y))
    for u,v in zip(y,expected_y):
      self.assertAlmostEqual(u,v,delta=0.2*v)
      
  def test_scaled_shifted(self):
    a = 4
    b = 6
    x = 1.5
    n = 21
    y = geometric_fineboth(a,b,n,x)
    expected_y = [0.0, 0.004, 0.01, 0.02, 0.036, 0.058, 0.092, 0.14, 0.22, \
                  0.33, 0.50, 0.67, 0.78, 0.86, 0.91, 0.94, 0.96, 0.98, \
                  0.99, 0.996, 1.0]
    expected_y = [y*(b-a) + a for y in expected_y]
    self.assertEqual(len(y),len(expected_y))
    for u,v in zip(y,expected_y):
      self.assertAlmostEqual(u,v,delta=0.2*(v-4))
      
  def test_even(self):
    a = 0
    b = 1
    x = 1.5
    n = 20
    y = geometric_fineboth(a,b,n,x)
    expected_y = [0.0, 0.004, 0.01, 0.02, 0.036, 0.058, 0.092, 0.14, 0.22, \
                  0.33, 0.67, 0.78, 0.86, 0.91, 0.94, 0.96, 0.98, 0.99, \
                  0.996, 1.0]
    self.assertEqual(len(y),len(expected_y))
    for u,v in zip(y,expected_y):
      self.assertAlmostEqual(u,v,delta=0.2*v)
      
  def test_x_equal_one(self):
    a = 0
    b = 1
    x = 1+1e-10 # uniform step
    n = 21
    y = geometric_fineboth(a,b,n,x)
    expected_y = list(np.arange(0,1.0+0.05/2,0.05))
    self.assertEqual(len(y),len(expected_y))
    for u,v in zip(y,expected_y):
      self.assertAlmostEqual(u,v,delta=0.2*v)


if __name__ == '__main__':
    unittest.main()  # Run all the tests