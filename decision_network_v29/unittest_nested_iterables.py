import unittest
from nested_iterables import flatten,flatten_enumerate,get_nested_item

class TestFlatten(unittest.TestCase):
  
  def test_basic(self):
    x = [1,[2,3],[4,[5,6]],7]
    expected_y = [1,2,3,4,5,6,7]
    y = flatten(x)
    self.assertEqual(y, expected_y)

  def test_empty(self):
    x = []
    expected_y = []
    y = flatten(x)
    self.assertEqual(y, expected_y)
    
  def test_nested_empty(self):
    x = [1,[2,3],[4,[]],7]
    expected_y = [1,2,3,4,7]
    y = flatten(x)
    self.assertEqual(y, expected_y)
    
  def test_double_nest(self):
    x = [1,[2,3],[[5,6]],7]
    expected_y = [1,2,3,5,6,7]
    y = flatten(x)
    self.assertEqual(y, expected_y)    
    
class TestFlattenEnumerate(unittest.TestCase):

  def test_basic(self):
    x = [1,[2,3],[4,[5,6]],7]
    expected_y = [1,2,3,4,5,6,7]    
    expected_iy = [[0],[1,0],[1,1],[2,0],[2,1,0],[2,1,1],[3]]
    expected_z = list(zip(expected_iy,expected_y))
    z = list(flatten_enumerate(x))
    self.assertEqual(z, expected_z)
    
  def test_empty(self):
    x = []
    expected_y = []
    expected_iy = []
    expected_z = list(zip(expected_iy,expected_y))
    z = list(flatten_enumerate(x))
    self.assertEqual(z, expected_z)
    
  def test_nested_empty(self):
    x = [1,[2,3],[4,[]],7]
    expected_y = [1,2,3,4,7]
    expected_iy = [[0],[1,0],[1,1],[2,0],[3]]
    expected_z = list(zip(expected_iy,expected_y))
    z = list(flatten_enumerate(x))
    self.assertEqual(z, expected_z)
    
  def test_double_nest(self):
    x = [1,[2,3],[[5,6]],7]
    expected_y = [1,2,3,5,6,7]
    expected_iy = [[0],[1,0],[1,1],[2,0,0],[2,0,1],[3]]
    expected_z = list(zip(expected_iy,expected_y))
    z = list(flatten_enumerate(x))
    self.assertEqual(z, expected_z)   
    
class GetNestedItems(unittest.TestCase):
  
  def test_basic(self):
    x = [1,[2,3],[4,[5,6]],7]
    iis = [2,1,1]
    expected_y = 6
    y = get_nested_item(x,iis)
    self.assertEqual(y, expected_y)
    
  def test_item_is_list(self):
    x = [1,[2,3],[4,[5,6]],7]
    iis = [2,1]
    expected_y = [5,6]
    y = get_nested_item(x,iis)
    self.assertEqual(y, expected_y)    

if __name__ == '__main__':
    unittest.main()  # Run all the tests