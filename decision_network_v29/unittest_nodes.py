import unittest
import numpy as np
import scipy.stats as stats
import math
import collections
import copy

from nodes import DAGNode, ValueDAGNode, DecisionNode, ChanceNode, \
                  StateNode, EvidenceNode, HiddenStateNode, UtilityNode

def quantile_counts(chance,num_draws):
  
  QUANTILE_BNDS = [0.005,0.995] # quantile bounds in which results must fall
  lower_count = stats.binom.ppf(QUANTILE_BNDS[0],num_draws,chance)
  upper_count = stats.binom.ppf(QUANTILE_BNDS[1],num_draws,chance)
  return (int(math.floor(lower_count)),int(math.ceil(upper_count)))
  
class TestDAGNode(unittest.TestCase):
    
  def test_add_child_updates_parent(self):
      
    a = DAGNode('a')
    b = DAGNode('b')
    c = DAGNode('c')
    d = DAGNode('d')
    e = DAGNode('e')
    
    a.add_child(b)
    self.assertTrue(a in b.parents)
    
    b.add_parent(c)
    self.assertEqual(set(b.parents),set((a,c)))
    
    b.remove_parent(a)
    self.assertEqual(set(a.children),set())
    
    d.extend_parents((a,b,c))
    self.assertEqual(set(d.parents),set((a,b,c)))
    self.assertTrue(d in a.children)
    self.assertTrue(d in b.children)
    self.assertTrue(d in c.children)
    
    e.add_child(a)
    e.extend_children((b,d))
    self.assertEqual(set(e.children),set((a,b,d)))
    self.assertTrue(e in a.parents)
    self.assertTrue(e in b.parents)
    self.assertTrue(e in d.parents)
    
  def test_clear_parents_children(self):
    
    a = DAGNode('a')
    b = DAGNode('b')
    c = DAGNode('c')
    d = DAGNode('d')
    e = DAGNode('e')
    f = DAGNode('f')
    
    a.extend_children((b,c,d))
    e.extend_children((b,c))
    b.add_child(f)
    
    self.assertEqual(set(a.children),set((b,c,d)))
    self.assertEqual(set(b.parents),set((a,e)))
    self.assertEqual(set(b.children),set((f,)))
    self.assertEqual(set(c.parents),set((a,e)))
    self.assertEqual(set(d.parents),set((a,)))
    self.assertEqual(set(e.children),set((b,c)))
    
    a.clear_children()
    
    self.assertEqual(set(a.children),set())
    self.assertEqual(set(b.parents),set((e,)))
    self.assertEqual(set(b.children),set((f,)))
    self.assertEqual(set(c.parents),set((e,)))
    self.assertEqual(set(d.parents),set())
    self.assertEqual(set(e.children),set((b,c)))
    
    b.clear_children()
    e.clear_children()
    
    self.assertEqual(set(a.children),set())
    self.assertEqual(set(b.children),set())
    self.assertEqual(set(c.children),set())
    self.assertEqual(set(d.children),set())
    self.assertEqual(set(e.children),set())
    
    a.extend_children((b,c,d))
    e.extend_children((b,c))
    b.add_child(f)
    
    b.clear_parents()
    
    self.assertEqual(set(a.children),set((c,d)))
    self.assertEqual(set(b.children),set((f,)))
    self.assertEqual(set(c.children),set())
    self.assertEqual(set(d.children),set())
    self.assertEqual(set(e.children),set((c,)))
    
    c.add_child(d)
    c.clear()
    
    self.assertEqual(set(a.children),set((d,)))
    self.assertEqual(set(b.children),set((f,)))
    self.assertEqual(set(c.children),set())
    self.assertEqual(set(d.children),set())
    self.assertEqual(set(e.children),set())
    
  def test_str_and_repr_basic(self):
    
    node = DAGNode("my node")
    self.assertEqual(str(node), "DAGNode(name='my node')")
    self.assertEqual(repr(node), "DAGNode(name='my node')")
    self.assertEqual(node.short_repr(), "my node")
    

class TestValueDAGNodeValue(unittest.TestCase):

  def test_round_trip_conversion_tuple_flat(self):

    node = ChanceNode('node',dim_sizes=(2,5,3))
    
    # value -> tuple
    node.value = 10
    x = node.value_tuple
    self.assertEqual(x,(0,3,1)) # big-endian, 0*(5*3) +  3*(3) + 1*(1) = 10
    
    # tuple -> value
    node.value_tuple = (1,3,1)
    y = node.value
    self.assertEqual(y,25)      # big-endian, 1*(5*3) +  3*(3) + 1*(1) = 25

  def test_single_dimension(self):
  
    node = ChanceNode('1D node',total_values=7)
    node.value = 4
    self.assertEqual(node.value_tuple,(4,))
    node.value_tuple = (5,)
    self.assertEqual(node.value,5)
    
  def test_invalid_tuple_length(self):
    
    node = ChanceNode('node',dim_sizes=(2, 5))
    with self.assertRaises(ValueError):
      node.value_tuple = (1, 2, 3) # too many coords
      
  def test_invalid_coord_range(self):
  
    node = ChanceNode('node',dim_sizes=(2,5))
    with self.assertRaises(ValueError):
      node.value_tuple = (2,0) # 2 invalid for first dimensions
      
  def test_short_repr_1d(self):
      
    node = ValueDAGNode('observer',total_values=5)
    node.value = 0
    self.assertEqual(node.short_repr(),'observer = 0')
    node.value = 3
    self.assertEqual(node.short_repr(),'observer = 3')
    
  def test_short_repr_multid(self):
      
    node = ValueDAGNode('states', dim_sizes=(2,5,3))
    node.value = 0
    self.assertEqual(node.short_repr(), "states = (0, 0, 0)")
    node.value = 10
    self.assertEqual(node.short_repr(), "states = (0, 3, 1)")
    
class TestValueDAGNodeForwardLinking(unittest.TestCase):

  def test_forward_link(self):

    node1 = ChanceNode('node1',total_values=3)
    node2 = ChanceNode('node2',total_values=3)
    node3 = ChanceNode('node3',total_values=3)
    node4 = ChanceNode('node4',total_values=3)

    node1.add_forward_link(node2)
    node1.add_forward_link(node3)
    node2.add_forward_link(node4)

    node1.value = 0
    node2.value = 0
    node3.value = 0
    node4.value = 0

    node1.value = 1
    self.assertTrue(node1.value == 1)
    self.assertTrue(node2.value == 1)
    self.assertTrue(node3.value == 1)
    self.assertTrue(node4.value == 1)

    node2.value = 2
    self.assertTrue(node1.value == 1)
    self.assertTrue(node2.value == 2)
    self.assertTrue(node3.value == 1)
    self.assertTrue(node4.value == 2)
    
  def test_forward_link_cycle_safety_strict(self):
    
    a = ValueDAGNode('a',total_values=2)
    b = ValueDAGNode('b',total_values=2)
    
    a.add_forward_link(b)
    b.add_forward_link(a)
    
    a.value = 0
    b.value = 0
    
    a.value = 1
    
    self.assertEqual(a.value,1)
    self.assertEqual(b.value,1)
    
    a.value = 0
    
    self.assertEqual(a.value,0)
    self.assertEqual(b.value,0)
    
  def test_forward_link_cycle_tri_safety_strict(self):
    
    a = ValueDAGNode('a',total_values=2)
    b = ValueDAGNode('b',total_values=2)
    c = ValueDAGNode('c',total_values=2)
    
    a.add_forward_link(b)
    b.add_forward_link(c)
    c.add_forward_link(a)
    
    a.value = 0
    b.value = 0
    c.value = 0
    
    a.value = 1
    
    self.assertEqual(a.value,1)
    self.assertEqual(b.value,1)
    self.assertEqual(c.value,1)
    
    a.value = 0
    
    self.assertEqual(a.value,0)
    self.assertEqual(b.value,0)
    self.assertEqual(c.value,0)


class TestDecisionNode(unittest.TestCase):
    
  def test_default_decision_type(self):
    node = DecisionNode('first',total_values=2)
    self.assertEqual(node.decision_type,"max")
    
  def test_explicit_decision_type(self):
    node1 = DecisionNode('first',total_values=2,decision_type="min")
    self.assertEqual(node1.decision_type, "min")
    
    node2 = DecisionNode('second',total_values=2,decision_type="max")
    self.assertEqual(node2.decision_type, "max")
    
    with self.assertRaises(ValueError):
      node3 = DecisionNode('third',total_values=2,decision_type="frodo")
      
  def test_value_setting(self):
    node = DecisionNode('first',total_values=5)
    node.value = 3
    self.assertEqual(node.value,3)
    
    with self.assertRaises(ValueError):
      node.value = 7
      
    with self.assertRaises(TypeError):
      node.value = 2.7
      
  def test_init_with_extra_kwargs(self):
    node = DecisionNode('first',dim_sizes=(5,2),value=2,decision_type="min")
    self.assertEqual(node.decision_type, "min")
    self.assertEqual(node.dim_sizes,(5,2))
    self.assertEqual(node.value,2)
    self.assertEqual(node.value_tuple,(1,0))
    
    
class TestChanceNode(unittest.TestCase):
    
  def test_init_with_extra_kwargs(self):
    node = ChanceNode('node',dim_sizes=(2,2),table=[[0.25,0.25,0.25,0.25],\
                                                    [0.92,0.03,0.05,0.00]])
    self.assertEqual(node.table[0][0],0.25)
    self.assertEqual(node.table[1][3],0.00)


class TestChanceNodeSPT(unittest.TestCase):

  def setUp(self):
    
    self.node = ChanceNode('node',total_values=3)
    self.spt_good = np.array((0.85,0.05,0.10))
    self.spt_badsum = np.array((0.85,0.10,0.10))
    self.spt_wrongsize = np.array((0.85,0.05,0.05,0.05))
    
    self.num_quantile_draws = 1000

  def test_table_verification_good(self):

    self.node.table = self.spt_good
    
  def test_table_verification_wrong_size(self):

    with self.assertRaises(ValueError):
      self.node.table = self.spt_wrongsize
    
  def test_table_verification_invalid_probabilities_init(self):

    with self.assertRaises(ValueError):
      self.node.table = self.spt_badsum

  def test_table_verification_invalid_probabilities_verify(self):

    self.node._table = self.spt_badsum # bypass property setter verification
    with self.assertRaises(ValueError):
      self.node.verify_table()

  def test_random_draws_quantile(self):

    self.node.table = self.spt_good

    count = collections.Counter()
    for _ in range(self.num_quantile_draws):
      self.node.set_random_value()
      count.update([self.node.value])
    
    print('\nCounts for set_random_value (SPT)')
    for i,x in enumerate(self.node.table):
      [a,b] = quantile_counts(x,self.num_quantile_draws)
      self.assertTrue(a <= count.get(i,0) <= b)
      print('{:6d} <= {:6d} <= {:6d}'.format(a,count.get(i,0),b))

  def test_get_probability(self):
    
    self.node.table = self.spt_good
    self.node.value = 1
    p = self.node.get_probability()
    self.assertEqual(p,0.05)
    
  def test_create_random_table(self):
    
    self.node._table = None # bypass property setter verification
    self.node.create_random_table()
    self.assertEqual(self.node._table.shape,(3,))
    
  def test_short_repr(self):
    
    self.node.table = self.spt_good
    self.assertEqual(self.node.short_repr(),\
                     "node = 0; table = [0.85 0.05 0.1 ]")


class TestChanceNodeCPT(unittest.TestCase):

  def setUp(self):
  
    self.cpt_good = np.zeros((2,3,2))
    self.cpt_good[..., 0] = [[0.1, 0.2, 0.3], [0.5, 0.4, 0.0]]
    self.cpt_good[..., 1] = [[0.9, 0.8, 0.7], [0.5, 0.6, 1.0]]
    
    self.cpt_badsum = np.zeros((2,3,2))
    self.cpt_badsum[..., 0] = [[0.1, 0.2, 0.3], [0.5, 0.4, 0.0]]
    self.cpt_badsum[..., 1] = [[0.9, 0.8, 0.7],[ 0.5, 0.65, 1.0]]
    
    self.cpt_wrongsize = np.zeros((2,2,2))
    self.cpt_wrongsize[..., 0] = [[0.1, 0.2], [0.4, 0.0]]
    self.cpt_wrongsize[..., 1] = [[0.9, 0.8], [0.6, 1.0]]
    
    self.cpt_wrongnumdims = np.zeros((4,2))
    self.cpt_wrongnumdims[..., 0] = [0.1, 0.2, 0.4, 0.0]
    self.cpt_wrongnumdims[..., 1] = [0.9, 0.8, 0.6, 1.0]
    
    self.parent1 = ChanceNode('parent1',total_values=2)
    self.parent2 = ChanceNode('parent2',total_values=3)
    self.node = ChanceNode('node',total_values=2)
    self.node.extend_parents((self.parent1, self.parent2))
    
    self.num_quantile_draws = 1000

  def test_table_verification_good(self):

    self.node.table = self.cpt_good
    self.node.verify_table()
    
  def test_table_verification_invalid_number_dimensions(self):
    
    self.node._table = self.cpt_wrongnumdims # bypass property setter veri.
    with self.assertRaises(ValueError):
      self.node.verify_table()
      
  def test_table_verification_wrong_size(self):

    self.node._table = self.cpt_wrongsize # bypass property setter verification
    with self.assertRaises(ValueError):
      self.node.verify_table()
      
  def test_table_verification_slice_sum_not_one(self):

    self.node._table = self.cpt_badsum # bypass property setter verification
    with self.assertRaises(ValueError) as context:
      self.node.verify_table()
      
  def test_random_draws_quantile(self):
    
    self.node.table = self.cpt_good
    self.parent1.value = 1
    self.parent2.value = 1
    
    count = collections.Counter()
    for _ in range(self.num_quantile_draws):
      self.node.set_random_value()
      count.update([self.node.value])
      
    print('\nCounts for set_random_value (CPT)')
    for i,x in enumerate(self.node.table[self.parent1.value,\
                                         self.parent2.value]):
      [a,b] = quantile_counts(x,self.num_quantile_draws)
      self.assertTrue(a <= count.get(i,0) <= b)
      print('{:6d} <= {:6d} <= {:6d}'.format(a,count.get(i,0),b))
      
  def test_get_probability(self):
      
    self.node.table = self.cpt_good
    self.parent1.value = 1
    self.parent2.value = 1
    self.node.value = 0
    p = self.node.get_probability()
    self.assertEqual(p,0.4)
    
  def test_create_random_table(self):
    
    self.node._table = None # bypass property setter verification
    self.node.create_random_table()
    self.assertEqual(self.node._table.shape,(2,3,2))

  def test_short_repr(self):
    
    self.node.table = [[0.1,0.9],[0.5,0.5]]
    self.assertEqual(self.node.short_repr(),\
                     "node = 0; table = [[0.1 0.9]\n [0.5 0.5]]")
                     

class TestStateEvidenceHiddenStateNode(unittest.TestCase):

  def setUp(self):
    self.cpt = np.zeros((2,2))
    self.cpt = [[0.1, 0.9], [0.5, 0.5]]
  
  def test_state_node(self):
    node = StateNode('node',total_values=2)
    node.table = self.cpt
    self.assertEqual(node.short_repr(),\
                     "node = 0; table = [[0.1 0.9]\n [0.5 0.5]] [STATE]")
    self.assertIsInstance(node, DAGNode)
    self.assertIsInstance(node,ValueDAGNode)
    self.assertIsInstance(node,ChanceNode)
    self.assertIsInstance(node,StateNode)
    self.assertNotIsInstance(node,EvidenceNode)
    self.assertNotIsInstance(node,HiddenStateNode)
    self.assertNotIsInstance(node,UtilityNode)
  
  def test_evidence_node(self):
    node = EvidenceNode('node',total_values=2)
    node.table = self.cpt
    self.assertEqual(node.short_repr(),\
                     "node = 0; table = [[0.1 0.9]\n [0.5 0.5]] [EVIDENCE]")
    self.assertIsInstance(node, DAGNode)
    self.assertIsInstance(node,ValueDAGNode)
    self.assertIsInstance(node,ChanceNode)
    self.assertIsInstance(node,EvidenceNode)
    self.assertNotIsInstance(node,StateNode)
    self.assertNotIsInstance(node,HiddenStateNode)
    self.assertNotIsInstance(node,UtilityNode)
  
  def test_hiddenstate_node(self):
    node = HiddenStateNode('node',total_values=2)
    node.table = self.cpt
    self.assertEqual(node.short_repr(),\
                     "node = 0; table = [[0.1 0.9]\n [0.5 0.5]] [HIDDEN]")
    self.assertIsInstance(node, DAGNode)
    self.assertIsInstance(node,ValueDAGNode)
    self.assertIsInstance(node,ChanceNode)
    self.assertIsInstance(node,HiddenStateNode)
    self.assertNotIsInstance(node,StateNode)
    self.assertNotIsInstance(node,EvidenceNode)
    self.assertNotIsInstance(node,UtilityNode)
  

class TestUtilityNode(unittest.TestCase):

  def setUp(self):
      
    self.ut_good = np.zeros((2,3))
    self.ut_good[0, ...] = [1, 2, 3]
    self.ut_good[1, ...] = [5, 4, 0]

    self.ut_wrongsize = np.zeros((2,4))
    self.ut_wrongsize[0, ...] = [1, 2, 3, 8]
    self.ut_wrongsize[1, ...] = [5, 4, 0, 8]
    
    self.ut_wrongnumdims = np.zeros((2,3,2))
    self.ut_wrongnumdims[0, ...] = [[1, 2], [3, 8], [1, 1]]
    self.ut_wrongnumdims[1, ...] = [[5, 4], [0, 0], [1, 2]]
    
    self.parent1 = ChanceNode('parent1',total_values=2)
    self.parent2 = ChanceNode('parent2',total_values=3)
    self.node = UtilityNode('node')
    self.node.extend_parents((self.parent1, self.parent2))

  def test_utility_table_verification_good(self):

    self.node._table = self.ut_good # bypass property setter verification
    self.node.verify_table()
    
  def test_utility_table_verification_invalid_number_dimensions(self):

    self.node._table = self.ut_wrongnumdims # bypass property setter veri.
    with self.assertRaises(ValueError):
      self.node.verify_table()
    
  def test_utility_table_verification_wrong_size(self):

    self.node._table = self.ut_wrongsize # bypass property setter verification
    with self.assertRaises(ValueError):
      self.node.verify_table()
  
  def test_get_utility(self):
    
    self.node.table = self.ut_good
    self.parent1.value = 1
    self.parent2.value = 1
    u = self.node.get_utility()
    self.assertEqual(u,4)
    
  def test_create_random_table(self):
    
    self.node._table = None # bypass property setter verification
    self.node.create_random_table()
    self.assertEqual(self.node._table.shape,(2,3))
    
  def test_short_repr(self):
    
    self.node.table = [[1,9],[5,5]]
    self.assertEqual(self.node.short_repr(),\
                     "node; table = [[1. 9.]\n [5. 5.]]")

    
if __name__ == '__main__':
  unittest.main()  # Run all the tests