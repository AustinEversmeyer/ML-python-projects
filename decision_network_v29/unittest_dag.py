import unittest
import os
import numpy as np
from itertools import product, combinations
from pathlib import Path

from dag import topological_sort_dfs, assign_positions_grid, dfs_reachable, \
                draw_directed_graph, DAG, DecisionNetwork

from nodes import ChanceNode, StateNode, HiddenStateNode, EvidenceNode, \
                  DecisionNode, UtilityNode, ValueDAGNode, DAGNode

class TestTopologicalSortDFS(unittest.TestCase):
  """Tests topogical_sort_dfs function"""
  
  def verify_order(self, order, edges):
    pos = dict((node, i) for i, node in enumerate(order))
    return all(pos[parent] < pos[child] for parent, child in edges)
  
  def test_tree(self):
      
    # tree
    edges = [(1,2),(1,3),(3,4),(3,5),(3,6),(4,7)]
    nodes = [1,2,3,4,5,6,7]
    order = topological_sort_dfs(nodes,edges)
    self.assertTrue(self.verify_order(order, edges))
    
    # tree with nodes in different order
    edges = [(1,2),(1,3),(3,4),(3,5),(3,6),(4,7)]
    nodes = [7,5,3,1,4,2,6]
    order = topological_sort_dfs(nodes,edges)
    self.assertTrue(self.verify_order(order, edges))
    
    # tree with edges in different order
    edges = [(3,4),(1,3),(3,5),(3,6),(1,2),(4,7)]
    nodes = [7,5,3,1,4,2,6]
    order = topological_sort_dfs(nodes,edges)
    self.assertTrue(self.verify_order(order, edges))
    
  def test_forest(self):
    
    # forest
    edges = [(3,4),(8,9),(1,3),(3,5),(0,8),(3,6),(1,2),(4,7)]
    nodes = [7,5,3,0,1,4,2,6,8]
    order = topological_sort_dfs(nodes,edges)
    self.assertTrue(self.verify_order(order, edges))
    
  def test_cycle(self):
    
    # cycle but not directed cycle, adds (1,6)
    edges = [(3,4),(1,3),(3,5),(3,6),(1,2),(4,7),(1,6)]
    nodes = [7,5,3,1,4,2,6]
    order = topological_sort_dfs(nodes,edges)
    self.assertTrue(self.verify_order(order, edges))
    
    # cycle
    edges = [(1,2),(2,3),(3,4),(4,5),(5,1)]
    nodes = [1,2,3,4,5]
    with self.assertRaises(ValueError):
      order = topological_sort_dfs(nodes,edges)
      
    # cycle plus edge
    edges = [(1,2),(2,3),(2,6),(3,4),(4,5),(5,1)]
    nodes = [6,1,2,3,4,5]
    with self.assertRaises(ValueError):
      order = topological_sort_dfs(nodes,edges)
      
    # cycle plus chord
    edges = [(1,2),(2,5),(2,3),(3,4),(4,5),(5,1)]
    nodes = [1,2,3,4,5]
    with self.assertRaises(ValueError):
      order = topological_sort_dfs(nodes,edges)
      
    # cycle plus chord induced
    edges = [(1,2),(2,6),(6,5),(2,3),(3,4),(4,5),(5,1)]
    nodes = [1,2,3,4,5,6]
    with self.assertRaises(ValueError):
      order = topological_sort_dfs(nodes,edges)
      
      
class TestAssignPositionsGrid(unittest.TestCase):
  """Tests assign positions grid function"""
  
  def test_one(self):
    
    nodes = [1,2,3,4,5]
    node2pos_init = {1:(0,0.1),2:(1,0),3:(1,1)}
    node2pos = assign_positions_grid(nodes, node2pos_init)
  
    for k,v in node2pos_init.items():
      self.assertEqual(v,node2pos[k])
    
    a = node2pos[2]
    b = node2pos[3]

    for x,y in [a,b]:
      self.assertEqual(x % 1,0)
      self.assertEqual(y % 1,0)
      self.assertLessEqual(abs(x) + abs(y), 2)
    
    for s,t in combinations(node2pos.values(), 2):
      d = abs(s[0] - t[0]) + abs(s[1] + t[1])
      self.assertGreater(d,0.8)
      
      
class TestDFSReachable(unittest.TestCase):
  """Tests dfs_reachable function"""
  
  def test_tree(self):
      
    # tree
    edges = [(1,2),(1,3),(3,4),(3,5),(3,6),(4,7)]
    nodes = [1,2,3,4,5,6,7]
    self.assertTrue(dfs_reachable(1,1,edges))
    self.assertTrue(dfs_reachable(1,4,edges))
    self.assertFalse(dfs_reachable(4,1,edges))
    self.assertFalse(dfs_reachable(4,5,edges))
    self.assertFalse(dfs_reachable(5,4,edges))
    
  def test_forest(self):
    
    # forest
    edges = [(3,4),(8,9),(1,3),(3,5),(0,8),(3,6),(1,2),(4,7)]
    nodes = [7,5,3,0,1,4,2,6,8]
    self.assertTrue(dfs_reachable(1,5,edges))
    self.assertFalse(dfs_reachable(5,1,edges))
    self.assertFalse(dfs_reachable(1,0,edges))
    self.assertFalse(dfs_reachable(0,1,edges))
    self.assertFalse(dfs_reachable(1,8,edges))
    self.assertFalse(dfs_reachable(8,1,edges))
    self.assertTrue(dfs_reachable(0,8,edges))
    
  def test_cyle_plus_edge(self):
    
    # cycle plus edge
    edges = [(1,2),(2,3),(3,4),(4,5),(5,1),(3,6)]
    for i,j in combinations([1,2,3,4,5,6],2):
      self.assertEqual(dfs_reachable(i,j,edges), i!=6)
      

class TestDrawDirectedGraph(unittest.TestCase):
  """Tests drawing of directed graph"""
  
  def test_korb_fever(self):
    
    nodes = ['flu','fever','therm','aspirin','feverlater','reaction','utility']
    edges = [('flu','fever'),
             ('fever','therm'),
             ('fever','feverlater'),
             ('aspirin','feverlater'),
             ('aspirin','reaction'),
             ('feverlater','utility'),
             ('reaction','utility')]
    temporal_edges = [('therm','aspirin')]
      
    edgestylecolors = [('solid','black')]*len(edges) + \
                       [('dashed','gray')]*len(temporal_edges)
      
      
    node2pos = {'flu':(0,2),
                'aspirin':(1,2),
                'reaction':(2,1.5),
                'fever':(0,1),
                'feverlater':(1,1),
                'utility':(2,0.5),
                'therm':(0,0)}
                    
    node2shape = {'flu':'ellipse',
                  'aspirin':'rectangle',
                  'reaction':'ellipse',
                  'fever':'ellipse',
                  'feverlater':'ellipse',
                  'utility':'diamond',
                  'therm':'ellipse'}
                  
    node2label = {'flu':'Flu',
                  'aspirin':'Aspirin',
                  'reaction':'Reaction',
                  'fever':'Fever',
                  'feverlater':'Fever Later',
                  'utility':'Utility',
                  'therm':'Therm'}
    
    path = Path('./data/korb_fever.png')
    path_str = str(path.resolve())
    if os.path.isfile(path_str):
      os.remove(path_str)
    
    self.assertFalse(path.exists(), f"File {path} should not exist") 
    draw_directed_graph(nodes,
                        edges + temporal_edges,
                        path_str,
                        node2pos = node2pos,
                        node2shape = node2shape,
                        node2label = node2label,
                        edgestylecolors = edgestylecolors)
    self.assertTrue(path.exists(), f"File {path} should exist") 

    
class TestDAG(unittest.TestCase):
  """Tests for DAG node/edge management, consistency, and cycle safety."""
  
  def setUp(self):
    self.a = DAGNode("A")
    self.b = DAGNode("B")
    self.c = DAGNode("C")
    self.d = DAGNode("D")
    self.e = DAGNode("E")
    
    self.flu = DAGNode('flu')
    self.fever = DAGNode('fever')
    self.therm = DAGNode('therm')
    self.aspirin = DAGNode('aspirin')
    self.feverlater = DAGNode('feverlater')
    self.reaction = DAGNode('reaction')
    self.utility = DAGNode('utility')
    
    self.korb_nodes = [self.flu,self.fever,self.therm,self.aspirin,
                       self.feverlater,self.reaction,self.utility]
    self.korb_edges = [(self.flu,self.fever),
                       (self.fever,self.therm),
                       (self.fever,self.feverlater),
                       (self.aspirin,self.feverlater),
                       (self.aspirin,self.reaction),
                       (self.feverlater,self.utility),
                       (self.reaction,self.utility)]
    self.korb_temporal_edges = [(self.therm,self.aspirin)]
    self.korb_fever_g = DAG("test", self.korb_nodes, self.korb_edges, 
                                    self.korb_temporal_edges)
      
    self.korb_node2pos = {self.flu:(0,2),
                          self.aspirin:(1,2),
                          self.reaction:(2,1.5),
                          self.fever:(0,1),
                          self.feverlater:(1,1),
                          self.utility:(2,0.5),
                          self.therm:(0,0)}
    
  def test_init_empty(self):
    g = DAG("empty")
    self.assertEqual(g.name, "empty")
    self.assertEqual(len(g.nodes), 0)
    self.assertEqual(len(g.nodes), 0)
    self.assertEqual(len(g.temporal_edges), 0)
    
  def test_init_simple(self):
    nodes = [self.a, self.b, self.c, self.d]
    edges = [(self.a, self.b), (self.b, self.c)]
    temporal_edges = [(self.c, self.d)]
    g = DAG('simple', nodes, edges, temporal_edges)
    
    self.assertEqual(len(g.nodes), 4)
    self.assertEqual(len(g.edges), 2)
    self.assertEqual(len(g.temporal_edges), 1)
    
  def test_init_edge_not_in_nodes(self):
    nodes = [self.a, self.b, self.d]
    edges = [(self.a, self.b), (self.b, self.c)]
    temporal_edges = [(self.c, self.d)]
    with self.assertRaises(ValueError):
      g = DAG('simple', nodes, edges, temporal_edges)
      
  def test_init_cycle(self):
    nodes = [self.a, self.b, self.c, self.d]
    edges = [(self.a, self.b), (self.b, self.c), (self.c, self.d)]
    temporal_edges = [(self.d, self.a)]
    with self.assertRaises(ValueError):
      g = DAG('simple', nodes, edges, temporal_edges)
      
  def test_init_dirty_node(self):
    nodes = [self.a, self.b, self.c, self.d]
    edges = [(self.a, self.b), (self.b, self.c)]
    temporal_edges = [(self.c, self.d)]
    self.a.add_child(self.b)
    with self.assertRaises(ValueError):
      g = DAG('simple', nodes, edges, temporal_edges)
      
  def test_set_nodes_simple(self):
    """Initialize then set"""
    nodes = [self.a, self.b, self.c, self.d]
    edges = [(self.a, self.b), (self.b, self.c)]
    temporal_edges = [(self.c, self.d)]
    g = DAG('simple', nodes, edges, temporal_edges)
    self.assertEqual(len(g.nodes), 4)
    self.assertEqual(len(g.edges), 2)
    self.assertEqual(len(g.temporal_edges), 1)
    
    self.a.clear()
    g.nodes = [self.a, self.e]
    self.assertEqual(len(g.nodes), 2)
    self.assertEqual(len(g.edges), 0)
    self.assertEqual(len(g.temporal_edges), 0)
    
  def test_set_dirty_node(self):
    nodes = [self.a, self.b, self.c, self.d]
    edges = [(self.a, self.b), (self.b, self.c)]
    temporal_edges = [(self.c, self.d)]
    g = DAG('simple', nodes, edges, temporal_edges)
    
    with self.assertRaises(ValueError):
      g.nodes = [self.a, self.e]
      
  def test_set_not_a_dagnode(self):
    nodes = [self.a, self.b, self.c, self.d]
    edges = [(self.a, self.b), (self.b, self.c)]
    temporal_edges = [(self.c, self.d)]
    g = DAG('simple', nodes, edges, temporal_edges)
    
    with self.assertRaises(TypeError):
      g.nodes = [self.e, 'good morning']
      
  def test_set_duplicate_node(self):
    nodes = [self.a, self.b, self.c, self.d]
    edges = [(self.a, self.b), (self.b, self.c)]
    temporal_edges = [(self.c, self.d)]
    g = DAG('simple', nodes, edges, temporal_edges)
    
    self.a.clear()
    with self.assertRaises(ValueError):
      g.nodes = [self.e, self.a, self.e]
      
  def test_set_edges_simple(self):
    nodes = [self.a, self.b, self.c, self.d]
    edges = [(self.a, self.b), (self.b, self.c)]
    temporal_edges = [(self.c, self.d)]
    g = DAG('simple', nodes, edges, temporal_edges)
    self.assertEqual(len(g.nodes), 4)
    self.assertEqual(len(g.edges), 2)
    self.assertEqual(len(g.temporal_edges), 1)
    
    g.edges = [(self.a, self.d)]
    self.assertEqual(len(g.nodes), 4)
    self.assertEqual(len(g.edges), 1)
    self.assertEqual(len(g.temporal_edges), 1)
    self.assertIn(self.a, self.d.parents)
    self.assertIn(self.d, self.a.children)
    
    self.a.clear()
    self.b.clear()
    self.c.clear()
    g.temporal_edges = [(self.a, self.b), (self.a, self.c)]
    self.assertEqual(len(g.nodes), 4)
    self.assertEqual(len(g.edges), 1)
    self.assertEqual(len(g.temporal_edges), 2)
    self.assertNotIn(self.a, self.d.parents)
    self.assertNotIn(self.d, self.a.children)
    
  def test_set_edges_not_in_nodes(self):
    nodes = [self.a, self.b, self.c, self.d]
    edges = [(self.a, self.b), (self.b, self.c)]
    temporal_edges = [(self.c, self.d)]
    g = DAG('simple', nodes, edges, temporal_edges)
    self.assertEqual(len(g.nodes), 4)
    self.assertEqual(len(g.edges), 2)
    self.assertEqual(len(g.temporal_edges), 1)
    
    with self.assertRaises(ValueError):
      g.edges = [(self.a, self.e)]
    
    with self.assertRaises(ValueError):
      g.temporal_edges = [(self.a, self.e)]
      
  def test_set_edges_cycle(self):
    nodes = [self.a, self.b, self.c, self.d]
    edges = [(self.a, self.b), (self.b, self.c)]
    temporal_edges = [(self.c, self.d)]
    g = DAG('simple', nodes, edges, temporal_edges)
    self.assertEqual(len(g.nodes), 4)
    self.assertEqual(len(g.edges), 2)
    self.assertEqual(len(g.temporal_edges), 1)
    
    with self.assertRaises(ValueError):
      g.edges = [(self.a, self.b), (self.b, self.c), (self.c, self.a)]
    
    with self.assertRaises(ValueError):
      g.temporal_edges = [(self.a, self.b), (self.b, self.c), (self.c, self.a)]
      
    g.edges = []
    g.temporal_edges = []
    
    g.edges = [(self.a, self.b), (self.b, self.c)]
    with self.assertRaises(ValueError):
      g.temporal_edges = [(self.c, self.a)]
      
    g.edges = []
    g.temporal_edges = []
    
    g.temporal_edges = [(self.a, self.b), (self.b, self.c)]
    with self.assertRaises(ValueError):
      g.edges = [(self.c, self.a)]
      
  def test_set_edges_duplicate(self):
    nodes = [self.a, self.b, self.c, self.d]
    edges = [(self.a, self.b), (self.b, self.c)]
    temporal_edges = [(self.c, self.d)]
    g = DAG('simple', nodes, edges, temporal_edges)
    self.assertEqual(len(g.nodes), 4)
    self.assertEqual(len(g.edges), 2)
    self.assertEqual(len(g.temporal_edges), 1)
    
    with self.assertRaises(ValueError):
      g.edges = [(self.a, self.b), (self.b, self.c), (self.a, self.b)]
    
    with self.assertRaises(ValueError):
      g.temporal_edges = [(self.a, self.b), (self.b, self.c), (self.a, self.b)]
    
  def test_add_node(self):
    g = DAG("test")
    g.add_node(self.a)
    self.assertEqual(len(g.nodes),1)
    self.assertIn(self.a, g.nodes)
    
    # duplicate node raises error
    with self.assertRaises(ValueError):
      g.add_node(self.a)
      
    # rejects dirty node
    self.b.add_parent(self.c)
    with self.assertRaises(ValueError):
      g.add_node(self.b)
      
    # rejects not a valid node
    with self.assertRaises(TypeError):
      g.add_node('not a valid node')
      
    # add another node
    self.b.clear()
    g.add_node(self.b)
    self.assertEqual(len(g.nodes),2)
    self.assertIn(self.b, g.nodes)
    
  def test_add_edge(self):
    nodes = [self.a, self.b, self.c, self.d]
    edges = [(self.a, self.b), (self.b, self.c)]
    temporal_edges = [(self.c, self.d)]
    g = DAG('simple', nodes, edges, temporal_edges)
    self.assertEqual(len(g.nodes), 4)
    self.assertEqual(len(g.edges), 2)
    self.assertEqual(len(g.temporal_edges), 1)
    
    # add edge
    g.add_edge(self.a, self.d)
    self.assertEqual(set(self.a.children), set((self.b, self.d)))
    
    # add temporal edge
    g.add_temporal_edge(self.a, self.c)
    self.assertEqual(set(self.a.children), set((self.b, self.d)))
    
    # duplicate edge
    with self.assertRaises(ValueError):
      g.add_edge(self.a, self.b)
    
    # duplicate temporal edge
    with self.assertRaises(ValueError):
      g.add_temporal_edge(self.c, self.d)
    
    for node in nodes:
      node.clear()
    g = DAG('simple', nodes, edges, temporal_edges)
    
    # edge is already a temporal edge
    with self.assertRaises(ValueError):
      g.add_edge(self.c, self.d)
    
    # temporal edge is already a normal edge
    with self.assertRaises(ValueError):
      g.add_temporal_edge(self.c, self.d)
      
    # add edge not in nodes
    with self.assertRaises(ValueError):
      g.add_edge(self.c, self.e)
    
    # add temporal edge not in nodes
    with self.assertRaises(ValueError):
      g.add_edge(self.c, self.e)
    
    nodes = [self.a, self.b, self.c, self.d]
    for node in nodes:
      node.clear()
    edges = [(self.a, self.b), (self.b, self.c)]
    temporal_edges = [(self.c, self.d)]
    g = DAG('simple', nodes, edges, temporal_edges)
    
    # add normal edge cycle
    with self.assertRaises(ValueError):
      g.add_edge(self.c, self.a)
    
    # add temporal edge cycle
    with self.assertRaises(ValueError):
      g.add_temporal_edge(self.d, self.e)
      g.add_temporal_edge(self.e, self.c)
      
  def test_set_nodes_clears_edges(self):
    
    # setting nodes clears edges
    g = DAG("test")
    g.add_node(self.a)
    g.add_node(self.b)
    g.add_edge(self.a, self.b)
    
    self.assertEqual(len(g.edges), 1)
    
    # adding node through add does not clear edges
    g.add_node(self.c)
    self.assertEqual(len(g.edges), 1)
    
    # assigning nodes does clear edges
    self.a.clear()
    self.b.clear()
    g.nodes = [self.a, self.b, self.c, self.d]
    self.assertEqual(len(g.edges), 0)
    
  def test_topological_order(self):
   
   topo_order = self.korb_fever_g.topological_order()
   self.assertTrue(set(topo_order),set(self.korb_fever_g.nodes))
   pos = dict((node, i) for i, node in enumerate(topo_order))
   for p, c in self.korb_fever_g.edges + self.korb_fever_g.temporal_edges:
     self.assertLess(pos[p],pos[c])
  
  def test_draw_graph(self):
    
    path = Path('./data/korb_fever2.png')
    path_str = str(path.resolve())
    if os.path.isfile(path_str):
      os.remove(path_str)

    self.assertFalse(path.exists(), f"File {path} should not exist") 
    self.korb_fever_g.draw_graph(path_str, self.korb_node2pos)
    self.assertTrue(path.exists(), f"File {path} should exist") 


class TestDecisionNetwork(unittest.TestCase):

  def setUp(self):

    self.flu        = HiddenStateNode('flu',
                                 total_values = 2,
                                 table = np.array([0.05,0.95]))
    self.fever      = HiddenStateNode('fever',
                                 total_values = 2,
                                 table=np.array([[0.95,0.05],[0.02,0.98]]))
    self.therm      = EvidenceNode('therm',
                                 total_values = 2,
                                 table=np.array([[0.90,0.10],[0.05,0.95]]))
    self.aspirin    = DecisionNode('aspirin',
                                 total_values = 2)
    self.feverlater = HiddenStateNode('feverlater',
                                 total_values = 2,
                                 table=np.array([[[0.05,0.95],[0.90,0.10]],
                                                 [[0.01,0.99],[0.02,0.98]]]))
    self.reaction   = HiddenStateNode('reaction',
                                 total_values = 2,
                                 table=np.array([[0.05,0.95],[0.00,1.00]]))
    self.utility    = UtilityNode('utility',
                                 table=np.array([[-50,-10],[-30,50]]))

    self.korb_nodes = [self.flu,self.fever,self.therm,self.aspirin,
                       self.feverlater,self.reaction,self.utility]
    self.korb_edges = [(self.flu,self.fever),
                       (self.fever,self.therm),
                       (self.fever,self.feverlater),
                       (self.aspirin,self.feverlater),
                       (self.aspirin,self.reaction),
                       (self.feverlater,self.utility),
                       (self.reaction,self.utility)]
    self.korb_temporal_edges = [(self.therm,self.aspirin)]
    self.korb_fever_g = DecisionNetwork("korb",
                                        nodes=self.korb_nodes,
                                        edges=self.korb_edges, 
                                        temporal_edges=self.korb_temporal_edges)
    self.korb_node2pos = {self.flu:(0,2),
                          self.aspirin:(1,2),
                          self.reaction:(2,1.5),
                          self.fever:(0,1),
                          self.feverlater:(1,1),
                          self.utility:(2,0.5),
                          self.therm:(0,0)}

  def test_korb_fever_probability(self):
    self.flu.value = 0
    self.fever.value = 0
    self.therm.value = 0
    self.aspirin.value = 0
    self.feverlater.value = 1
    self.reaction.value = 1
    expected_p = 0.05*0.95*0.90*0.95*0.95
    expected_u = 50
    expected_wu = expected_p*expected_u
    self.assertAlmostEqual(self.korb_fever_g.get_probability(),
                           expected_p, delta=0.001)
    self.assertAlmostEqual(self.korb_fever_g.get_utility(),
                           expected_u, delta=0.001)
    self.assertAlmostEqual(self.korb_fever_g.get_weightedutility(),
                           expected_wu, delta=0.001)
    
  def test_create_random_tables(self):
    a = HiddenStateNode('a', total_values = 2)
    b = HiddenStateNode('b', total_values = 3)
    c = EvidenceNode('c', total_values = 2)
    nodes = [a, b, c]
    edges = [(a, c), (b, c)]
    g = DecisionNetwork("test",nodes = nodes, edges = edges)
    self.assertEqual(c.table.size, 0)
    g.create_random_tables()
    self.assertEqual(c.table.shape, (2, 3, 2))
    c.verify_table()
    
  def test_temporal_decision_order(self):
    
    order = self.korb_fever_g.temporal_decision_order()
    self.assertEqual(len(order),6) # nodes except utility
    self.assertEqual(order[0],self.therm)
    self.assertEqual(order[1],self.aspirin)
    
  def test_draw_graph(self):
    self.korb_fever_g.draw_graph('./data/korb_fever_with_node_type.png',
                                 self.korb_node2pos)
                                 
  def test_str(self):
    expected_str = """DecisionNetwork(name='korb',nodes=7,edges=7,
                      temporal_edges=1)"""
    expected_repr = expected_str
    def rmv(x):
      return x.replace(' ','').replace('\n','')
    self.assertEqual(rmv(str(self.korb_fever_g)),rmv(expected_str))
    self.assertEqual(rmv(repr(self.korb_fever_g)),rmv(expected_repr))
    
  def test_node_class_to_list(self):
    d = self.korb_fever_g.node_class_to_list() # class -> list
    self.assertEqual(set(d[DecisionNode]),set((self.aspirin,)))
    self.assertEqual(set(d[UtilityNode]),set((self.utility,)))
    self.assertEqual(set(d[ChanceNode]),set((self.flu,self.fever,self.therm,\
                                             self.feverlater,self.reaction)))
    self.assertEqual(set(d[HiddenStateNode]),set((self.flu,\
                                  self.fever,self.feverlater,self.reaction)))
    self.assertEqual(set(d[EvidenceNode]),set((self.therm,)))
    self.assertEqual(set(d[StateNode]),set())
    
    x = DecisionNode('decision1',total_values=2)
    y = EvidenceNode('evidence1',total_values=2)
    self.korb_fever_g.add_node(x)
    self.assertEqual(set(d[DecisionNode]),set((self.aspirin,x)))
    self.korb_fever_g.add_node(y)
    self.assertEqual(set(d[EvidenceNode]),set((self.therm,y)))
    
  def test_get_nodes(self):
    nodes = self.korb_fever_g.nodes
    self.assertEqual(set(nodes),set((self.flu,self.fever,self.therm,\
                                     self.feverlater,self.reaction,\
                                     self.aspirin,self.utility)))
                                     
  def test_set_nodes(self):
    x = ValueDAGNode('x',total_values=2)
    y = ValueDAGNode('y',total_values=2)
    self.korb_fever_g.nodes = [x,y]
    self.assertEqual(len(self.korb_fever_g.nodes),2)

  def test_set_nodes_not_dag_node(self):
    x = ValueDAGNode('temp',total_values=2)
    y = 'not a dag node'
    with self.assertRaises(TypeError):
      self.korb_fever_g.nodes = [x,y]
  
  def test_set_nodes_dirty(self):
    x = ValueDAGNode('x',total_values=2)
    y = ValueDAGNode('y',total_values=2)
    x.add_child(y)
    with self.assertRaises(ValueError):
      self.korb_fever_g.nodes = [x,y]
  
  def test_add_nodes(self):
    x = ValueDAGNode('x',total_values=2)
    y = ValueDAGNode('y',total_values=2)
    self.korb_fever_g.add_node(x)
    self.korb_fever_g.add_node(y)
    self.assertEqual(len(self.korb_fever_g.nodes),9)
    
  def test_add_node_not_dag_node(self):
    y = 'not a dag node'
    with self.assertRaises(TypeError):
      self.korb_fever_g.add_node(y)
      
  def test_add_node_dirty(self):
    x = ValueDAGNode('x',total_values=2)
    y = ValueDAGNode('y',total_values=2)
    x.add_child(y)
    with self.assertRaises(ValueError):
      self.korb_fever_g.add_node(x)
      

if __name__ == '__main__':
  unittest.main()  # Run all the tests