from __future__ import annotations
import sys
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Polygon
from collections import defaultdict
from heapq import heappush, heappop
from typing import List, Iterable, Tuple, Any

from nodes import DAGNode, ChanceNode, UtilityNode, DecisionNode, StateNode, \
                  HiddenStateNode, EvidenceNode

def topological_sort_dfs(
    nodes: Iterable[Any],
    edges: Iterable[Tuple[Any,Any]]) -> List[T]:
  """
  topological sort graph so parents appear left of children
  """
  
  children_map: dict[Any, list[Any]] = defaultdict(list)
  for parent, child in edges:
    children_map[parent].append(child)

  visited: set[Any] = set()
  rec_stack: set[Any] = set()
  order: list[Any] = []
  
  def dfs(node: Any) -> None:
    visited.add(node)
    rec_stack.add(node)
    
    for child in children_map[node]:
      if child not in visited:
        dfs(child)
      elif child in rec_stack:
        raise ValueError(f"Cycle detected involving node {child}")
    order.append(node)
    rec_stack.remove(node)

    
  for node in nodes[::-1]:
    if node not in visited:
      dfs(node)
  
  order.reverse()
  return order
  
def assign_positions_grid(
    nodes: Iterable[Any],
    node2pos: Dict[Any, Tuple[float, float]] | None =None,
    min_distance:float = 1.2) -> Dict[Any, Tuple[float, float]]:
  """
  Assign grid positions to nodes, preferring positions close to (0,0) and
  avoiding already occupied areas (minimum Euclidean distance).
  
  Uses a priority queue (heap) to expand outward in Manhattan distance order.
  
  Args:
    nodes: Iterable of nodes to place
    node2posL Optional existing dict node -> (x, y) position
    min_distance: Minimum Euclidean distance between any two nodes
  
  Returns:
    Dict mapping each node to its assigned (x, y) position (float coords.)
    
  Notes:
    If no free spint is found (very unlikely), falls back to a horizontal line.
  """
  
  # Default to empty dict if not provided
  node2pos = node2pos.copy() if node2pos is not None else {}
  
  # Already placed positions (as tuples for fast lookup)
  placed: set[Tuple[float, float]] = set(node2pos.values())
  
  # Priority queue: (manhattan_distance from origin, x, y)
  pq: list[Tuple[int, int, int]] = []
  seen: set[Tuple[int, int]] = set()

  # Start exploration from origin
  heappush(pq, (0, 0, 0))  # (dist, x, y)
  seen.add((0, 0))
  
  # Neighbors to expand (king's movement: 8 directions)
  neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]

  for node in set(nodes) - set(node2pos.keys()):
    placed_found = False
    
    while pq:
      _, x, y = heappop(pq)
      pos = (float(x), float(y))

      # Check distance to all placed points
      if all(math.hypot(x - px, y - py) >= min_distance for px, py in placed):
        node2pos[node] = pos
        placed.add(pos)
        placed_found = True
        break

      # Enqueue neighbors (diamond expansion)
      for dx, dy in neighbors:
        nx, ny = x + dx, y + dy
        if (nx, ny) not in seen:
          seen.add((nx, ny))
          new_dist = abs(nx) + abs(ny)
          heappush(pq, (new_dist, nx, ny))
    
    if not placed_found:
      # Fallback: place on a horizontal line to the right
      x_fallback = len(node2pos) * 2.0
      node2pos[node] = (x_fallback, 0.0)
      placed.add((x_fallback, 0.0))
  
  return node2pos
  
def dfs_reachable(
    root: Any,
    target: Any,
    edges: Iterable[Tuple[Any, Any]]
) -> bool:
  """
  Return True if there is a path from root to target.
  
  Uses DFS with a visited set to avoid revisiting nodes.
  
  Args:
    root  : starting node
    target: node we want to reach
    edges : Iterable of (parent, child) directed edges
  
  Returns:
    bool: True if path exists, False otherwise
  """
  if root is None or target is None:
    return False
  if root == target:
    return True
  
  # Build adjacency list once (children map)
  children_map: dict[T, list[T]] = defaultdict(list)
  for parent, child in edges:
    children_map[parent].append(child)
  
  visited: set[Any] = set()
  
  def dfs(current: Any) -> bool:
    if current in visited:
      return False
    visited.add(current)
    
    for child in children_map[current]:
      if child == target:
        return True
      if dfs(child):
        return True
    return False
  
  return dfs(root)
  
def draw_directed_graph(
    nodes: Iterable[Any],
    edges: Iterable[Tuple[Any, Any]],
    png_path,
    node2pos: Optional[Dict[Any, Tuple[float, float]]] = None,
    node2shape: Optional[Dict[Any, str]] = None,
    node2label: Optional[Dict[Any, str]] = None,
    edgestylecolors: Iterable[Tuple[str, str]] = None,
) -> None:
  """
  Draw a directed graph with custom node shapes and per-edge styles/colors.
  Saves the result as a PNG file.
  
  Args:
    nodes          : Iterable of nodes
    edges          : Iterable of (parent, child) pairs
    png_path       : Path to save the PNG file
    node2pos       : Optional dict node -> (x, y) position
    node2shape     : Optional dict node -> shape name ('rectangle', 'ellipse',
                     'diamond')
    node2label     : Optional dict node -> label string
    edgestylecolors: Iterable of (style, color) e.g.
                     ('dashed', 'gray') or ('solid', 'black')
  """
  # Constants
  SCALE        = 5.0
  FONT_SIZE    = 6
  SHAPE_SCALE  = 1.45
  DEFAULT_FACE = 'white'
  DEFAULT_EDGE = 'black'
  DEFAULT_LW   = 1.0
  ARROW_MARGIN = 0.5
  DEFAULT_ESTYLE = 'solid'
  DEFAULT_ECOLOR = 'black'

  # Defaults
  node2pos = node2pos or {}
  node2shape = node2shape or {}
  node2label = node2label or {}
  edgestylecolors = edgestylecolors or \
                                    [(DEFAULT_ESTYLE,DEFAULT_ECOLOR)]*len(edges)

  # Shape dimensions (width, height)
  shape_dims = {
    'rectangle': (1.0 * SHAPE_SCALE, 0.5 * SHAPE_SCALE),
    'ellipse':   (1.1 * SHAPE_SCALE, 0.8 * SHAPE_SCALE),
    'diamond':   (0.5 * SHAPE_SCALE, 0.5 * SHAPE_SCALE),  # half-diagonal
   }

  # Assign missing positions
  node2pos = assign_positions_grid(nodes, node2pos.copy())
  
  # Scale all positions
  node2pos = {node: np.array(pos)*SCALE for node,pos in node2pos.items()}
  
  # Compute bounds after scaling
  if node2pos:
    positions = np.array(list(node2pos.values()))
    min_x, max_x = positions[:,0].min(), positions[:,0].max()
    min_y, max_y = positions[:,1].min(), positions[:,1].max()
    bounds = [min_x, max_x, min_y, max_y]
  else:
    bounds = [-SCALE, SCALE, -SCALE, SCALE]
      
  # prepare labels, shapes
  labels = {n: node2label.get(n, str(n)) for n in nodes}
  shapes = {n: node2shape.get(n, 'rectangle') for n in nodes}
  
  # prepare edgelist and styles, colors
  edgelist = []
  edge_styles = []
  edge_colors = []
  
  for edge, edgestylecolor in zip(edges, edgestylecolors):
    style, color = edgestylecolor
    edgelist.append(edge)
    edge_styles.append(style)
    edge_colors.append(color)
  
  # create graph
  G = nx.DiGraph()
  G.add_nodes_from(nodes)
  G.add_edges_from(edges)
  
  # plot setup
  fig, ax = plt.subplots()
  ax.set_xlim(bounds[0] - 1.5, bounds[1] + 1.5)
  ax.set_ylim(bounds[2] - 1.5, bounds[3] + 1.5)

  # dynamic arrow margin
  pt_per_unit = ax.transData.transform((1, 0))[0] - \
                ax.transData.transform((0, 0))[0]
  margin = pt_per_unit * ARROW_MARGIN
 
  # Draw the edges with arrows, applying margin to pull back arrowheads
  for i,edge in enumerate(edgelist):
      
    rad = 0.0 if edge_styles[i] == 'solid' else 0.2

    nx.draw_networkx_edges(
        G,
        pos=node2pos,
        edgelist=[edge],
        ax=ax,
        arrows=True,
        arrowstyle='->',
        arrowsize=10,
        connectionstyle=f"arc3,rad={rad}",
        node_size=0,
        min_target_margin=margin,
        style=edge_styles[i],
        edge_color=edge_colors[i],
        width=1.5,
        alpha=0.9,
    )

  # draw custom node shapes 
  for node in G.nodes():
    x, y = node2pos[node]
    shape_name = shapes[node]
    w, h = shape_dims.get(shape_name, (1.0, 1.0))

    if shape_name == 'ellipse':
      ax.add_patch(
        Ellipse((x, y), w, h,
        facecolor=DEFAULT_FACE, edgecolor=DEFAULT_EDGE,linewidth=DEFAULT_LW))
    elif shape_name == 'diamond':
      points = [(x, y+h), (x+w, y), (x, y-h), (x-w, y)]  # top - clockwise
      ax.add_patch(
        Polygon(points, 
        facecolor=DEFAULT_FACE, edgecolor=DEFAULT_EDGE,linewidth=DEFAULT_LW))
    else:
      ax.add_patch(
        Rectangle((x - w/2, y - h/2), w, h,
        facecolor=DEFAULT_FACE, edgecolor=DEFAULT_EDGE,linewidth=DEFAULT_LW))

  # labels
  nx.draw_networkx_labels(G, pos=node2pos, labels=labels,
                          ax=ax, font_weight='normal', font_size=FONT_SIZE)

  plt.axis('off')
  plt.savefig(png_path, bbox_inches='tight', dpi=150)
  plt.close()
  

class DAG:
  """
  A class to represent a directed acyclic graph (DAG).
  
  Attributes:
  -----------
  name            : str, the name of the graph
  _nodes          : list of DAGNode
  _edges          : list of (parent,child) DAGNode tuples
  _temporal_edges : list of (parent,child) DAGNode tuples
  
  Properties:
  -----------
  nodes          : list of all nodes
  edges          : list of all directed edges (parent, child)
  temporal_edges : list of all directed edges (parent, child)
  
  Notes:
  ------
  - the parents, children properties of nodes are used for the "normal" edges,
    not the temporal edges
  """
    
  def __init__(self,
               name:str,
               nodes: Iterable[DAGNode] = (),
               edges: Iterable[Tuple[DAGNode,DAGNode]] = (),
               temporal_edges: Iterable[Tuple[DAGNode,DAGNode]] = ()):
    """
    Constructor for DAG
        
    Parameters:
    -----------
    name           : str, the name of the node
    nodes          : tuple of DAGNode
    edges          : tuple of two-tuples (DAGNode,DAGNode) of directed edges
    temporal_edges : tuple of two-tuples (DAGNode,DAGNode) of directed edges
    """
    self.name = name
    self._nodes: List[DAGNode] = []
    self._edges: List[Tuple[DAGNode, DAGNode]] = []
    self._temporal_edges: List[Tuple[DAGNode, DAGNode]] = []
    
    # Use setters for validation
    self.nodes = nodes
    self.edges = edges
    self.temporal_edges = temporal_edges
    
  @property
  def nodes(self) -> List[DAGNode]:
    return list(self._nodes)
    
  @nodes.setter
  def nodes(self, new_nodes: Iterable[DAGNode]):
    """
    Replace all nodes. New nodes must be clean (no parents/children).
    Existing node links are cleared.
    """
    new_nodes = list(new_nodes)
    
    if any(not isinstance(node,DAGNode) for node in new_nodes):
      raise TypeError("""Nodes must be DAGNodes""")
    
    if any(node.parents or node.children for node in new_nodes):
      raise ValueError("""Nodes must have no parents or children when added to 
                          a graph""")
                          
    if len(set(new_nodes)) != len(new_nodes):
      raise ValueError("""Nodes must be unique""")

    self._nodes = new_nodes
    self._edges = [] # edges are no longer valid
    self._temporal_edges = [] # temporal edges are no longer valid
    
  @property
  def edges(self) -> List[Tuple[DAGNode, DAGNode]]:
    return list(self._edges)
    
  @edges.setter
  def edges(self, new_edges: Iterable[DAGNode, DAGNode]) -> None:
    self._set_edges(new_edges, is_temporal=False)
    
  @property
  def temporal_edges(self) -> List[Tuple[DAGNode, DAGNode]]:
    return list(self._temporal_edges)
    
  @temporal_edges.setter
  def temporal_edges(self, new_edges) -> None:
    self._set_edges(new_edges, is_temporal=True)
    
  def _set_edges(self,
                 new_edges: Iterable[Tuple[DAGNode, DAGNode]],
                 is_temporal: bool = True) -> None:
    """
    Shared logical for setting either normal or temporal edges.
    
    Replace all edges. Ensures all endpoints are in nodes and updates node
    parent/child lists bidirectionally. Checks for cycles.
    """
    new_edges = list(new_edges)
    
    # Validate endpoints
    node_set = set(self._nodes)
    if any(p not in node_set or c not in node_set for p, c in new_edges):
      raise ValueError("All edge endpoints must be in the graph's nodes")
      
    # Check for duplicates
    if len(set(new_edges)) < len(new_edges):
      raise ValueError("Duplicate edges are not allowed.")
    
    # Validate acyclic
    temp_edges = new_edges + \
                          (self._edges if is_temporal else self._temporal_edges)
    order = topological_sort_dfs(self.nodes, temp_edges)
    if len(order) != len(self.nodes):
      raise ValueError("""Proposed edges would create a cycle 
                          (incomplete topological order).""")
    pos = {node: i for i, node in enumerate(order)}
    for p, c in new_edges:
      if pos[p] >= pos[c]:
        raise ValueError(f"Cycle via edge {p.name} -> {c.name}")
    
    if is_temporal:
      self._temporal_edges = new_edges
    else:
      # Clear existing links
      for node in self._nodes:
        node.clear_children()
        node.clear_parents()
    
      # Add new edges with bidirectional consistency
      self._edges = []
      for parent, child in new_edges:
        parent.add_child(child) # parent added through property setter
        self._edges.append((parent, child))
      
  def add_node(self, node: DAGNode) -> None:
    """Add a single node (must be clean)."""
    if not isinstance(node, DAGNode):
      raise TypeError("Node must be a DAGNode")
    if node.parents or node.children:
      raise ValueError("New node must have no existing parents or children")
    if node in self._nodes:
      raise ValueError("Node already exists.")
    self._nodes.append(node)
      
  def _would_create_cycle(self, parent: DAGNode, child:DAGNode) -> bool:
    """
    Return True if adding parent -> child would create a cycle.
    (i.e. if there is already a path from child to parent)
    """
    return dfs_reachable(child, parent, self._edges + self._temporal_edges)

  def add_edge(self, parent: DAGNode, child: DAGNode) -> None:
    self._add_edge(parent, child, is_temporal = False)

  def add_temporal_edge(self, parent: DAGNode, child: DAGNode) -> None:
    self._add_edge(parent, child, is_temporal = True)
      
  def _add_edge(self,
                parent: DAGNode,
                child: DAGNode,
                is_temporal:bool = True
                ) -> None:
    """Add a single directed edge with bidirectional update."""

    edge_str = "temporal_edge" if is_temporal else "edge"
    
    if parent not in self._nodes or child not in self._nodes:
      raise ValueError(f"Added edge {parent.name}-{child.name} "
                       f"endpoints not contained in nodes")
    
    if (parent,child) in (self._temporal_edges + self._edges):
      raise ValueError(f"""{edge_str} {parent.name} -> {child.name}
                        already exists""")
    
    if self._would_create_cycle(parent, child):
      raise ValueError(
        f"Adding {edge_str} {parent.name} -> {child.name} would create a cycle "
        f"(path already exists from {child.name} to {parent.name})"
      )
    
    if is_temporal:
      self._temporal_edges.append((parent,child))
    else:
      parent.add_child(child) # parent added through property setter
      self._edges.append((parent,child))
      
  def topological_order(self) -> List[DAGNode]:
    """
    Return a list of nodes in topological order (parents before children).
    """
    return topological_sort_dfs(self.nodes, self._edges + self._temporal_edges)
    
  def draw_graph(self,
                 png_path: str,
                 node2pos: dict[Any,Tuple[float,float]] | None = None) -> None:
    """
    Draw the graph and save as PNG.
    
    Args:
      png_path: Path where the PNG file will be saved/
      node2pos: Optional dict mapping nodes to (x, y) positions.
                If omitted or partial, missing positions are auto-assigned.
    """
    # Prepare labels
    node2label = {n: n.name for n in self._nodes}
    
    # Draw the combined graph
    draw_directed_graph(
      self._nodes,
      self._edges + self._temporal_edges,
      png_path,
      node2label=node2label,
      node2pos=node2pos,
      edgestylecolors=[('solid', 'black')]*len(self._edges) + \
                      [('dashed', 'gray')]*len(self._temporal_edges)
      )

  def __str__(self) -> str:
    return f"""DAG(name={self.name!r}, nodes={len(self.nodes)}, 
               edges={len(self.edges)},
               temporal_edges={len(self.temporal_edges)})"""
               
  def __repr__(self) -> str:
    return self.__str__()
    
class DecisionNetwork(DAG):
    
  def __init__(self,name: str, /, **kwargs):
    
    # super __init__ will call this @nodes.setter, @edges.setter, and
    # @temporal_edges.setter
    self._decision_nodes = []
    self._chance_nodes = []
    self._state_nodes = []
    self._evidence_nodes = []
    self._hiddenstate_nodes = []
    self._utility_nodes = []
    super().__init__(name, **kwargs)

  def node_class_to_list(self):
    return {
      DecisionNode    : self._decision_nodes,
      ChanceNode      : self._chance_nodes,
      StateNode       : self._state_nodes,
      EvidenceNode    : self._evidence_nodes,
      HiddenStateNode : self._hiddenstate_nodes,
      UtilityNode     : self._utility_nodes,
    }

  @property
  def nodes(self) -> List[DAGNode]:
    return list(self._nodes)

  @nodes.setter
  def nodes(self, new_nodes: Iterable[DAGNode]) -> None:
    """
    Replace all nodes. New nodes must be clean (no parents/children).
    Existing node links are cleared.
    """
    DAG.nodes.fset(self, new_nodes)

    for node_class, node_list in self.node_class_to_list().items():
      node_list.clear()
      node_list.extend([n for n in new_nodes if isinstance(n,node_class)])
    
  def add_node(self, node: DAGNode) -> None:
    """Add a single node (must be clean)."""
    super().add_node(node)
    
    for node_class, node_list in self.node_class_to_list().items():
      if isinstance(node, node_class):
        node_list.append(node)

  def draw_graph(self,png_path,node2pos=None):
    """
    Draw and save graph.
    """
    
    node2label = {n: n.name for n in self._nodes}
    
    node2shape = defaultdict(lambda: 'rectangle')
    for node in self._nodes:
      if isinstance(node,ChanceNode): # Chance is super of State, Evi, Hidden
        node2shape[node] = 'ellipse'
      elif isinstance(node,UtilityNode):
        node2shape[node] = 'diamond'
      else:
        node2shape[node] = 'rectangle'

    draw_directed_graph(self._nodes, self._edges + self._temporal_edges,
                      png_path, node2label=node2label,node2pos=node2pos,
                      node2shape=node2shape,
                      edgestylecolors=[('solid', 'black')]*len(self._edges) + \
                      [('dashed', 'gray')]*len(self._temporal_edges))
                        
  def temporal_decision_order(self):
    """
    Return nodes in order for temporal decisions. Excludes utility nodes. 
    Parents of temporal edges must precede their children. Hidden state nodes
    must follow all decisions.
    """
    
    nodes = [n for n in self._nodes if not isinstance(n,UtilityNode)]
    nodes = topological_sort_dfs(nodes,self._temporal_edges)
    
    N = len(nodes)
    temporal_children = defaultdict(lambda:[])
    for p,c in self._temporal_edges:
      temporal_children[p] = temporal_children[p] + [c]
    
    def should_swap(a,b):
      if b in temporal_children[a]:
        return False
      elif isinstance(b,HiddenStateNode) and isinstance(a,DecisionNode):
        return False
      return True
    
    for i in range(N-2):
      for j in range(N-i-1):
        a = nodes[j]
        b = nodes[j+1]
        if should_swap(a,b):
          nodes[j:j+2] = [b,a]
    
    self.verify_nodes_decision_order(nodes)
    
    return nodes
    
  def verify_nodes_decision_order(self,ns):

    temporal_children = defaultdict(lambda:[])
    for p,c in self._temporal_edges:
      temporal_children[p] = temporal_children[p] + [c]
    
    def check_order(a,b):
      if a in temporal_children[b]:
        return False
      if isinstance(a,HiddenStateNode) and isinstance(b,DecisionNode):
        return False
      return True
    
    for i in range(len(ns)):
      for j in range(i+1,len(ns)):
        valid = check_order(ns[i],ns[j])
        if not valid:
          print(ns[i].name,ns[j].name)
          raise Exception('Invalid decision order for nodes')
    
  def get_probability(self):
    
    if not self._chance_nodes:
      return 1.0
    
    prob = 1.0
    for node in self._chance_nodes:
      p = node.get_probability()
      prob *= p
      
    return prob
    
  def get_utility(self):
    
    if not self._utility_nodes:
      return 0.0
    
    total = 0.0
    for node in self._utility_nodes:
      u = node.get_utility()
      total += u
    
    return total

  def get_weightedutility(self):

    return self.get_probability() * self.get_utility()
    
  def create_random_tables(self):
    
    for x in self._chance_nodes + self._utility_nodes:
      x.create_random_table()
    
  def verify_tables(self):

    missv = [nd for nd in self._chance_nodes if not hasattr(nd,'verify_table')]
    if missv:
      raise TypeError(f"These nodes lack a 'verify_table' method: {missv}")
    for nd in self._chance_nodes:
      nd.verify_table()

    missv = [nd for nd in self._utility_nodes if not hasattr(nd,'verify_table')]
    if missv:
      raise TypeError(f"These nodes lack a 'verify_table' method: {missv}")
    for nd in self._utility_nodes:
      nd.verify_table()

    
  def __str__(self) -> str:
    return f"""DecisionNetwork(name={self.name!r}, nodes={len(self.nodes)}, 
               edges={len(self.edges)}, 
               temporal_edges={len(self.temporal_edges)})"""
               
  def __repr__(self) -> str:
    return self.__str__()
    

def main():

  pass

if __name__ == '__main__':
  sys.exit(main())
