import unittest
import numpy as np
import itertools
from itertools import product
from math import prod
import time


from nodes import DAGNode, ValueDAGNode, DecisionNode, ChanceNode, \
                  StateNode, EvidenceNode, HiddenStateNode, UtilityNode
from dag import DAG, DecisionNetwork
from iterate import MultiDimIterator, MaskedMultiDimIterator, IteratorReducer

class TestMultiDimIterator(unittest.TestCase):
    
  def test_simple(self):
    
    it = MultiDimIterator((2,3,2))
    vals = []
    for idx in it:
      vals.append(idx)
    
    self.assertEqual(len(vals),2*3*2)
    
    self.assertEqual(vals[0],(0,0,0))
    self.assertEqual(vals[1],(0,0,1))
    self.assertEqual(vals[2],(0,1,0))
    self.assertEqual(vals[3],(0,1,1))
    self.assertEqual(vals[4],(0,2,0))
    self.assertEqual(vals[5],(0,2,1))
    self.assertEqual(vals[6],(1,0,0))
    self.assertEqual(vals[7],(1,0,1))
    self.assertEqual(vals[8],(1,1,0))
    self.assertEqual(vals[9],(1,1,1))
    self.assertEqual(vals[10],(1,2,0))
    self.assertEqual(vals[11],(1,2,1))
  
    vals = []
    for idx in it:
      vals.append(idx)
    self.assertEqual(len(vals),0)
    
    vals = []
    it.reset()
    self.assertEqual(len(it),12)
    for idx in it:
      vals.append(idx)
    self.assertEqual(len(vals),12)
    
    self.assertEqual(len(it),0)
    
  def test_not_a_tuple(self):
      
    with self.assertRaises(TypeError):
      it = MultiDimIterator()
      
    with self.assertRaises(TypeError):
      it = MultiDimIterator('not a tuple')
  
  def test_empty_tuple(self):
  
    with self.assertRaises(ValueError):
      it = MultiDimIterator(tuple())
      
  def test_zero_dim(self):
    
    with self.assertRaises(ValueError):
      it = MultiDimIterator((3,0,2))
    
  def test_repr(self):
     
    it = MultiDimIterator((2,3,2))
    expected_repr = "MultiDimIterator(shape=(2, 3, 2))"
    self.assertEqual(repr(it), expected_repr)
    
class TestMaskedMultiDimIterator(unittest.TestCase):
  
  def setUp(self):
    self.shape = (2,3,2)
    self.all_coords = [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(0,2,0),(0,2,1),
                       (1,0,0),(1,0,1),(1,1,0),(1,1,1),(1,2,0),(1,2,1)]
    self.all_inners = [0,1,0,1,0,1,0,1,0,1,0,1]
    self.all_prefixs = [0,0,1,1,2,2,3,3,4,4,5,5]
    self.all_flats = [0,1,2,3,4,5,6,7,8,9,10,11]
    self.mask = [True,True,True,True,True,False,False,True,True,True,True,True]
    self.it_fpi = MaskedMultiDimIterator(self.shape, 
                                     mask=self.mask, next_type='fpi')
    self.it_coord = MaskedMultiDimIterator(self.shape, 
                                     mask=self.mask, next_type='coord')
    self.it_flat = MaskedMultiDimIterator(self.shape, 
                                     mask=self.mask, next_type='flat')
                                  
  def test_simple(self):
    
    coords = []
    for idx in self.it_coord:
      coords.append(idx)
      
    fpis = []
    for idx in self.it_fpi:
      fpis.append(idx)
      
    flats = []
    for idx in self.it_flat:
      flats.append(idx)
    
    expected_coords = [x for x,y in zip(self.all_coords,self.mask) if y]
    expected_fpis = [(w,x,y) for w,x,y,z in 
         zip(self.all_flats,self.all_prefixs,self.all_inners,self.mask) if z]
    expected_flats = [x for x,y in zip(self.all_flats,self.mask) if y]
    
    self.assertEqual(coords,expected_coords)
    self.assertEqual(fpis,expected_fpis)
    self.assertEqual(flats,expected_flats)
    
  def test_init(self):
    
    # wrong input type for shape
    with self.assertRaises(TypeError):
      self.it = MaskedMultiDimIterator('none')
    
    # shape has a zero
    with self.assertRaises(ValueError):
      self.it = MaskedMultiDimIterator((3,0,3))
      
    # invalid next_type
    with self.assertRaises(ValueError):
      self.it = MaskedMultiDimIterator(self.shape, next_type='flatt')
    
    # no mask provided (defaults to all true)
    self.it = MaskedMultiDimIterator(self.shape, next_type='flat')
    self.assertEqual(len(self.it),12)
    
    # mask is wrong size
    with self.assertRaises(ValueError):
      self.it = MaskedMultiDimIterator(self.shape, 
                                       mask=[True,False,True], next_type='flat')
    
  def test_len(self):
    
    self.assertEqual(len(self.it_coord),10)
    self.assertEqual(len(self.it_fpi),10)
    self.assertEqual(len(self.it_flat),10)
    
    for idx in self.it_coord:
      pass
      
    for idx in self.it_fpi:
      pass
      
    for idx in self.it_flat:
      pass
      
    self.assertEqual(len(self.it_coord),0)
    self.assertEqual(len(self.it_fpi),0)
    self.assertEqual(len(self.it_flat),0)
    
    self.it_coord.reset()
    self.it_fpi.reset()
    self.it_flat.reset()
    
    self.assertEqual(len(self.it_coord),10)
    self.assertEqual(len(self.it_fpi),10)
    self.assertEqual(len(self.it_flat),10)
    
  def test_repr(self):
    
    expected_repr = ("MaskedMultiDimIterator(shape=(2, 3, 2), masked=10/12, "
                     "next_type=flat)")
    self.assertEqual(repr(self.it_flat),expected_repr)
    
  def test_reset(self):
    
    expected_coords = [x for x,y in zip(self.all_coords,self.mask) if y]
    
    coords = []
    for idx in self.it_coord:
      coords.append(idx)
    self.assertEqual(coords,expected_coords)
    self.it_coord.reset()
    for idx in self.it_coord:
      coords.append(idx)
    self.assertEqual(coords,expected_coords + expected_coords)
    
  def test_flat_prefix_inner_idx(self):

    fpis = []
    fpis2 = []
    for idx in self.it_fpi:
      fpis.append(idx)
      fpis2.append((self.it_fpi.flat_idx,
                    self.it_fpi.flat_prefix_idx,
                    self.it_fpi.inner_idx))
    self.assertEqual(fpis,fpis2)
    
  def test_coord_idx(self):
    
    coords = []
    coords2 = []
    for idx in self.it_coord:
      coords.append(idx)
      coords2.append(self.it_coord.coord_idx)
    self.assertEqual(coords,coords2)
    
    
class TestIteratorReducer(unittest.TestCase):
    
  def setUp(self):
      
    self.groups = ((2, 'sum'), (3, 'max'), (2, 'sum'))
    self.values = [1, 4, 2, 2, 3, 4, 5, 1, 3, 2, 1, 7]
    self.masks = [True, True, True, True, True, False,
                  True, True, True, True, True, False]

  def test_simple_docstring(self):
      
    ir = IteratorReducer(self.groups, self.values)
    vs,ds = ir.reduce()
    self.assertEqual(vs,np.array(15))    # value
    self.assertEqual(tuple(ds),(2, 2))   # decisions (max is last for each)
    
    groups = ((2, 'none'), (3, 'max'), (2, 'sum'))
    ir = IteratorReducer(groups, self.values)
    vs,ds = ir.reduce()
    self.assertEqual(tuple(vs), (7, 8))  # value
    self.assertEqual(tuple(ds), (2, 2))  # decisions (max is last for each)
    
    groups = ((2, 'none'), (3, 'none'), (2, 'sum'))
    ir = IteratorReducer(groups, self.values)
    vs,ds = ir.reduce()
    self.assertEqual(tuple(vs), (5, 4, 7, 6, 5, 8))  # value
    self.assertEqual(ds, None)                       # no decisions
    
  def test_simple_docstring_mask(self):
      
    ir = IteratorReducer(self.groups, self.values, self.masks)
    vs,ds = ir.reduce()
    self.assertEqual(vs,np.array(11))    # value
    self.assertEqual(tuple(ds),(0, 0))   # decisions (max is last for each)
    
    groups = ((2, 'none'), (3, 'max'), (2, 'sum'))
    ir = IteratorReducer(groups, self.values, self.masks)
    vs,ds = ir.reduce()
    self.assertEqual(tuple(vs), (5, 6))  # value
    self.assertEqual(tuple(ds), (0, 0))  # decisions (max is last for each)
    
    groups = ((2, 'none'), (3, 'none'), (2, 'sum'))
    ir = IteratorReducer(groups, self.values, self.masks)
    vs,ds = ir.reduce()
    self.assertEqual(tuple(vs), (5, 4, 3, 6, 5, 1))  # value
    self.assertEqual(ds, None)                       # no decisions
    
  def test_single_reduction_sum(self):
    
    groups = [(3, 'none'), (2, 'none'), (3, 'sum')]
    values   = [1,1,0,2,2,0,3,3,0,4,4,0,5,5,0,6,6,0]
    mask     = [1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,0,1]
    expected = [2,0,6,8,0,6]
    values1  = [1,1,0,2,2,0,3,3,0,4,4,0,5,5,0,7,7,0]
    expected1 = [2,0,6,8,0,7]
    ir = IteratorReducer(groups, values, mask_flat = mask)

    result,_ = ir.reduce()
    result1,_ = ir.reduce(values1)

    self.assertEqual(tuple(expected),tuple(result))
    self.assertEqual(tuple(expected1),tuple(result1))
    
  def test_single_reduction_max(self):
    
    groups = [(3, 'none'), (2, 'none'), (3, 'max')]
    values   = [1,1,0,2,2,0,3,3,0,4,4,0,5,5,0,6,6,0]
    mask     = [1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,0,1]
    expected = [1,-np.inf,3,4,-np.inf,6]
    ir = IteratorReducer(groups, values, mask_flat = mask)

    result,_ = ir.reduce()

    self.assertEqual(tuple(expected),tuple(result))
    
  def test_double_reduction_max_sum(self):

    groups = [(3, 'none'), (2, 'max'), (3, 'sum')]
    values   = [1,1,0,2,2,0,3,3,0,4,4,0,5,5,0,6,6,0]
    mask     = [1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,0,1]
    expected = [2,8,6]
    ir = IteratorReducer(groups, values, mask)
    result,_ = ir.reduce()
    self.assertEqual(tuple(expected),tuple(result))
    
  def test_double_reduction_sum_sum(self):
    
    groups = [(3, 'none'), (2, 'sum'), (3, 'sum')]
    values   = [1,1,0,2,2,0,3,3,0,4,4,0,5,5,0,6,6,0]
    mask     = [1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,0,1]
    expected = [2,14,6]
    ir = IteratorReducer(groups, values, mask)
    result,_ = ir.reduce()
    self.assertEqual(tuple(expected),tuple(result))
    
  def test_double_reduction_sum_max(self):

    groups = [(3, 'none'), (2, 'sum'), (3, 'max')]
    values   = [1,1,0,2,2,0,3,3,0,4,4,0,5,5,0,6,6,0]
    mask     = [1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,0,1]
    expected = [1,7,6]
    ir = IteratorReducer(groups, values, mask)
    result,_ = ir.reduce()
    self.assertEqual(tuple(expected),tuple(result))
    
  def test_korb_fever(self):
    
    flu        = HiddenStateNode('flu', total_values=2,
                                 table=np.array([0.05,0.95]))
    fever      = HiddenStateNode('fever', total_values=2,
                                  table=np.array([[0.95,0.05],[0.02,0.98]]))
    therm      = EvidenceNode('therm', total_values=2,
                                  table=np.array([[0.90,0.10],[0.05,0.95]]))
    aspirin    = DecisionNode('aspirin', total_values=2)
    feverlater = HiddenStateNode('feverlater', total_values=2,
                                  table=np.array([[[0.05,0.95],[0.90,0.10]],
                                                  [[0.01,0.99],[0.02,0.98]]]))
    reaction   = HiddenStateNode('reaction', total_values=2,
                                  table=np.array([[0.05,0.95],[0.00,1.00]]))
    utility = UtilityNode('utility',table=np.array([[-50,-10],[-30,50]]))
                                
    nodez = (flu,fever,therm,aspirin,feverlater,reaction,utility)
    edges = ((flu,fever),(fever,therm),(fever,feverlater),
             (aspirin,feverlater),(aspirin,reaction),(feverlater,utility),
             (reaction,utility))
    temporal_edges = ((therm,aspirin),)
    
    graph = DecisionNetwork('korb_fever',
                            nodes=nodez,
                            edges=edges,
                            temporal_edges=temporal_edges)
    graph.verify_tables()
    graph.draw_graph('./data/korb_fever.png',
                     node2pos={flu:(0,2),aspirin:(1,2),reaction:(2,1.5),\
                               fever:(0,1),feverlater:(1,1),utility:(2,0.5),\
                               therm:(0,0)})
                               
    # determine which nodes to reduce/iterate over (all but utility)
    node2reduce = {flu:'sum', fever:'sum', therm: 'sum', aspirin: 'max',\
                   feverlater: 'sum', reaction: 'sum'}
    decision_order = graph.temporal_decision_order()
    reducer = IteratorReducer([(node.total_values,node2reduce[node]) 
                                                 for node in decision_order])
                                                 
    # determine the weighted network utility for each combination of indices
    it = MaskedMultiDimIterator([node.total_values for node in decision_order],
                                next_type = "coord")

    wutils, probs = [], []
    for coord in it:
      for node, y in zip(decision_order, coord):
        node.value = y
      probs.append(graph.get_probability())
      wutils.append(graph.get_weightedutility())
    
    vs, ds = reducer.reduce(wutils)
    print('\nExpected Utility Take Aspirin')
    print('Expected Utility = {:8.4f}'.format(vs[0]))
    print('Decisions Therm = 0 (T)  -> Aspirin {:8d}'.format(ds[0]))
    print('Decisions Therm = 1 (F)  -> Aspirin {:8d}'.format(ds[1]))
    print(repr(reducer))
    self.assertEqual(tuple(ds),(0,1))
    
    # apply a mask to low probabilty nodes
    masks = [p > 0.001 for p in probs]
    reducer = IteratorReducer([(node.total_values,node2reduce[node]) 
                                 for node in decision_order], mask_flat = masks)
    
    it.reset()
    wutils, probs = [], []
    for coord in it:
      for node, y in zip(decision_order, coord):
        node.value = y
      probs.append(graph.get_probability())
      wutils.append(graph.get_weightedutility())
    
    vs, ds = reducer.reduce(wutils)
    print('\nExpected Utility Take Aspirin')
    print('Expected Utility = {:8.4f}'.format(vs[0]))
    print('Decisions Therm = 0 (T)  -> Aspirin {:8d}'.format(ds[0]))
    print('Decisions Therm = 1 (F)  -> Aspirin {:8d}'.format(ds[1]))
    print(repr(reducer))
    self.assertEqual(tuple(ds),(0,1))

    # determine expected utilities for all cases of therm x aspirin
    nodes_matrix = [therm, aspirin]
    nodes_loop = [flu, fever, feverlater, reaction] # reduce via sum
    nodes_all = nodes_matrix + nodes_loop
    
    it_matrix = MaskedMultiDimIterator(
               [node.total_values for node in nodes_matrix],next_type = "coord")
    it_loop = MaskedMultiDimIterator(
               [node.total_values for node in nodes_loop],next_type = "coord")
    reducer = IteratorReducer([(node.total_values,'sum') 
                                                        for node in nodes_loop])
    
    eus = np.zeros(it_matrix._shape)
    for coord_matrix in it_matrix:

      probs = []
      wutils = []

      # set fixed values for the matrix
      for node, y in zip(nodes_matrix, coord_matrix):
        node.value = y
      
      for coord_loop in it_loop:
        for node, y in zip(nodes_loop, coord_loop):
          node.value = y
        probs.append(graph.get_probability())
        wutils.append(graph.get_weightedutility())
      
      wu,_ = reducer.reduce(wutils)
      eus[coord_matrix] = wu.item()/sum(probs)
      it_loop.reset()
    
    self.assertAlmostEqual(eus[0,0],44.12,delta=0.01)
    self.assertAlmostEqual(eus[0,1],19.13,delta=0.01)
    self.assertAlmostEqual(eus[1,0],45.40,delta=0.01)
    self.assertAlmostEqual(eus[1,1],48.40,delta=0.01)

    
  def test_korb_sequential_decision(self):
    
    inspect   = DecisionNode('inspect', total_values=2)
    buy       = DecisionNode('buy', total_values=2)
    condition = HiddenStateNode('condition', total_values=2,
                                table=np.array([0.7,0.3]))
    report    = EvidenceNode('report', total_values=3,
                          table=np.array([[[0.95,0.05,0.00],[0.10,0.90,0.00]],
                                          [[0.00,0.00,1.00],[0.00,0.00,1.00]]]))
    u         = UtilityNode('u',table=np.array([-600,0]))
    v         = UtilityNode('v',table=np.array([[5000,0],[-3000,0]]))
                                
    nodez = (inspect,buy,condition,report,u,v)
    edges = ((inspect,u),(inspect,report),
             (condition,report),(condition,v),
             (buy,v))
    temporal_edges = ((inspect,buy),(report,buy),(inspect,report))
    
    graph = DecisionNetwork('graph',nodes=nodez,edges=edges,
                            temporal_edges=temporal_edges)
    graph.draw_graph('./data/korb_house.png',
                     node2pos={inspect:(0,1),u:(0,0),condition:(1,1),\
                               report:(1,0),buy:(1.5,2),v:(2,1)})
    graph.verify_tables()
    
    # determine which nodes to reduce/iterate over (all but utility)
    node2reduce = {inspect:'max', buy:'max', condition:'sum', report: 'sum'}
    decision_order = graph.temporal_decision_order()
    
    reducer = IteratorReducer([(node.total_values,node2reduce[node]) 
                                                 for node in decision_order])
    
    # determine the weighted network utility for each combination of indices
    it = MaskedMultiDimIterator([node.total_values for node in decision_order],
                                next_type = "coord")
    wutils, probs = [], []
    for coord in it:
      for node, y in zip(decision_order, coord):
        node.value = y
      probs.append(graph.get_probability())
      wutils.append(graph.get_weightedutility())
    
    vs, ds = reducer.reduce(wutils)
    
    print('\nExpected Utility Buy House Sequential Decisions')
    print('Expected Utility = {:8.4f}'.format(vs[0]))
    print('Decisions        = {:8d}'.format(ds[0]))
    self.assertAlmostEqual(vs[0],2635,delta=0.1)
    self.assertEqual(ds[0],0)
    
  def test_large(self):
    
    NUM_DEC = 5
    NUM_STA = 32
    NUM_EVI = 243
    dec_a = StateNode('dec(t-1)', total_values=NUM_DEC)
    dec_b = DecisionNode('dec(t)', total_values=NUM_DEC)
    sta_a = StateNode('sta(t-1)', total_values=NUM_STA)
    sta_b = HiddenStateNode('sta(t)', total_values=NUM_STA)
    sta_c = HiddenStateNode('sta(t+1)', total_values=NUM_STA)
    evi_b = EvidenceNode('evi(t)', total_values=NUM_EVI)
    uti_c = UtilityNode('utility')
    
    nodez = [dec_a,dec_b,sta_a,sta_b,sta_c,evi_b,uti_c]
    edges = [(dec_a,evi_b),
             (sta_a,sta_b),
             (dec_b,uti_c),
             (sta_b,sta_c),(sta_b,evi_b),
             (sta_c,uti_c)]
    temporal_edges = [(evi_b,dec_b)]
             
    graph = DecisionNetwork('graph',nodes=nodez,edges=edges,
                            temporal_edges=temporal_edges)
    graph.draw_graph('./data/large.png',
                     node2pos={dec_a:(0,2),dec_b:(1,2),
                               sta_a:(0,1),sta_b:(1,1),sta_c:(2,1),
                               evi_b:(1,0),uti_c:(2,0)})
    graph.create_random_tables()
    graph.verify_tables()
    
    node2reduce = {dec_b:'max', sta_b:'sum', sta_c:'sum', evi_b: 'sum'}
    decision_order = [node for node in graph.temporal_decision_order() if not
                      isinstance(node, StateNode)]
    reducer = IteratorReducer([(node.total_values,node2reduce[node]) 
                                                 for node in decision_order])

    dec_a.value = 0 # fixed value
    sta_a.value = 0 # fixed value

    # determine the weighted network utility for each combination of indices
    it = MaskedMultiDimIterator([node.total_values for node in decision_order],
                                next_type = "coord")
    wutils, probs = [], []
    for coord in it:
      for node, y in zip(decision_order, coord):
        node.value = y
      probs.append(graph.get_probability())
      wutils.append(graph.get_weightedutility())
    
    NUM_RUNS = 10
    
    start = time.perf_counter()
    for _ in range(NUM_RUNS):
    
      reduced_wutils,decisions = reducer.reduce(wutils)
    elapsed = time.perf_counter() - start
    
    print('')
    print('Average reduce run-time for length {:8d}: {:18.8f} ms'.\
          format(len(wutils),1e3*elapsed/NUM_RUNS))
    print('')

    
  def test_repr(self):
    ir = IteratorReducer(self.groups, self.values, self.masks)
    self.assertEqual(repr(ir),
      "IteratorReducer(shape=(2, 3, 2), masked=10/12, reductions=sum max sum)")
    
  def test_len(self):
    ir = IteratorReducer(self.groups, self.values, self.masks)
    self.assertEqual(len(ir),12)


if __name__ == '__main__':
  unittest.main()  # Run all the tests