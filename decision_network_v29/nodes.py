from __future__ import annotations
import sys
import numpy as np
from typing import Tuple, Optional, Union, Sequence, List
from collections.abc import Iterable
from numbers import Integral
import cpt_utils

class DAGNode:
  """
  Base class for nodes in a Directed Acyclic Graph (DAG).
  
  Attributes:
  -----------
  name        : str, the name of the node
  parents     : read-only view of parent nodes (DAGNode)
  children    : read-only view of parent nodes (DAGNode)
  
  Properties:
  -----------
  None
  
  Methods:
  --------
  __init__, add_child, add_parent, remove_child, remove_parent, extend_children,
  extend_parents, clear_children, clear_parents
  """
    
  def __init__(self, name: str):
    """
    Constructor for DAGNode
        
    Args:
    -----------
    name       : str, the name of the node
    """
    self.name: str = name
    self._parents: list['DAGNode'] = []
    self._children: list['DAGNode'] = []
    
  @property
  def parents(self) -> List['DAGNode']:
    return list(self._parents)
  
  @property
  def children(self) -> List['DAGNode']:
    return list(self._children)
    
  def add_child(self, child: 'DAGNode') -> None:
    if child is self:
      raise ValueError("Cannot add self as its own child in DAG (self-loop)")
    if child not in self._children:
      self._children.append(child)
    if self not in child.parents:
      child._parents.append(self)
  
  def add_parent(self, parent: 'DAGNode') -> None:
    if parent is self:
      raise ValueError("Cannot add self as its own parent in DAG (self-loop)")
    if parent not in self._parents:
      self._parents.append(parent)
    if self not in parent._children:
      parent._children.append(self)
      
  def remove_child(self, child: 'DAGNode') -> None:
    if not isinstance(child, DAGNode):
      raise TypeError(f"Expected DAGNode, got {type(child).__name__}")
    if child in self._children:
      self._children.remove(child)
    if self in child._parents:
      child._parents.remove(self)
      
  def remove_parent(self, parent: 'DAGNode') -> None:
    if not isinstance(parent, DAGNode):
      raise TypeError(f"Expected DAGNode, got {type(parent).__name__}")
    if parent in self._parents:
      self._parents.remove(parent)
    if self in parent._children:
      parent._children.remove(self)
      
  def extend_children(self, children: Iterable['DAGNode']) -> None:
    """
    Add multiple child nodes at once, ensuring bidirectional links.
    
    Args:
      children: Iterable of DAGNode instances to add as children
    """
    if children is None:
      return
    
    for child in children:
      self.add_child(child)
      
  def extend_parents(self, parents: Iterable['DAGNode']) -> None:
    """
    Add multiple parents at once by calling add_parent for each valid item.
    
    Args:
      parents: any iterable of DAGNode objects
    """
    if parents is None:
      return
    
    for parent in parents:
      self.add_parent(parent)
      
  def clear_children(self) -> None:
    """
    Remove all children of this node and break all corresponding parent links
    in the child nodes.
    """
    current_children = list(self.children)
    
    for child in current_children:
      if self in child._parents:
        child._parents.remove(self)
    
    self._children.clear()
  
  def clear_parents(self) -> None:
    """
    Remove all parents of this node and break all corresponding child links
    in the parent nodes.
    """
    
    current_parents = list(self._parents)
    
    for parent in current_parents:
      if self in parent._children:
        parent._children.remove(self)
    
    self._parents.clear()
    
  def clear(self) -> None:
    """
    Clears parents and children.
    """
    self.clear_children()
    self.clear_parents()
    
  def __str__(self) -> str:
    return f"{self.__class__.__name__}(name={self.name!r})"
    
  def __repr__(self) -> str:
    return f"{self.__class__.__name__}(name={self.name!r})"
  
  def short_repr(self) -> str:
    """Short string used in graph visualization, logs, etc."""
    return self.name


class ValueDAGNode(DAGNode):
  """
  DAG node that holds a discrete value integer.
  
  Can represent:
  - single discrete variable with N possible values (0..N-1)
  - multi-dimensional discrete variable

  Attributes:
  -----------
  _total_values    : integer, number of allowed values
  _value           : integer, the values of the node in 0 ... _total_values - 1,
                     -1 for not set
  _dim_sizes       : tuple of integer, node may represent multiple dimensions,
                     e.g. for multiple nodes; e.g. if _dim_szs = (3,5,2) then
                     _value = 15, represents (1,2,1) 1*10 + 2*2 + 1 = 15
  _dim_strides     : tuple of integer, used for calculations of multiple
                     dimension values; e.g. if _dim_szs = (3,5,2) then 
                     _dim_coefs = (10,2,1)
  _forward_link_nodes : list of nodes, when value of this node is changed, these
                        nodes are updated to the same value
  _updating           : boolean, true while forward linking is performed to
                        avoid circular recursion

  Properties:
  -----------
  total_values     : integer, number of allowed values; relies on private
                     attributes _total_values
  value            : integer, the value of the node; relies on private
                     attributes _value, _value_indicator
  dim_sizes        : tuple of integer, size of dimensions, if node has multiple
                     dimensions or represents multiple nodes
  value_tuple      : tuple of integer, read or get value via dimensional tuple;
                     this is for multiple dimensions, e.g. (3,5,2)

  Methods:
  --------
  __init__         : class constructor
  """

  def __init__(self,
               name: str,
               total_values: Optional[int] = None,
               dim_sizes: Optional[Sequence[int]] = None,
               value: Optional[int] = None):
    """
    Constructor for ValueDAGNode
        
    Args:
    -----------
    name           : identifier
    total_values   : number of possible values (for 1D case)
    dim_sizes      : tuple/list of dimension sizes (for multi-D case)
    value          : initial value (will be validated)
    """
    super().__init__(name)

    self._value: int = 0
    self._dim_sizes: Tuple[int, ...] = ()
    self._dim_strides: Tuple[int, ...] = ()
    self._total_values: int = total_values
    self._forward_link_nodes: list['ValueDAGNode'] = []
    self._updating = False # recursion guard
    
    if dim_sizes is not None:
      self.dim_sizes = tuple(dim_sizes) # also sets total_values & strides
      if total_values != None and (total_values != self._total_values):
        raise ValueError(
          f"Conflict for node {name!r}: "
          f"total_values={total_values} but "
          f"product(dim_sizes)={self._total_values}")
    elif total_values is not None:
      if total_values < 1:
        raise ValueError(f'total_values must be >= 1, got {total_values}')
      self._total_values = total_values
      self._dim_sizes = (total_values,)
      self._dim_strides = (1,)
    else:
      raise ValueError('Must provide either total_values or dim_sizes')

    if value is not None:
      self.value = value # users setter for validation

  @property
  def value(self) -> int:
    return self._value
    
  @value.setter
  def value(self, val: int) -> None:
    if not isinstance(val, int):
      raise TypeError(f"value must be int, got {type(val).__name__}")
    if not (0 <= val < self._total_values):
      raise ValueError(
        f"Value {val} out of range for node {self.name!r} "
        f"(valid: 0 <= x < {self._total_values})")
    
    if self._value == val:
      return # no change, no propagation
      
    if self._updating:
      return # already updating, skip (avoids infinite recursion)
      
    self._updating = True
    try:
      self._value = val
      
      # Propagate to forward-linked nodes
      for node in self._forward_link_nodes:
        node.value = val
    finally:
      self._updating = False

  def add_forward_link(self, node: 'ValueDAGNode') -> None:
    """Add a node that should be updated when this node's value changes."""
    if not isinstance(node, ValueDAGNode):
      raise TypeError("Can only link to other ValueDAGNode instances")
    if node is self:
      raise ValueError("Cannot create self-loop in forward links")
    if node not in self._forward_link_nodes:
      self._forward_link_nodes.append(node)
  
  def remove_forward_link(self, node: 'ValueDAGNode') -> None:
    """Remove a previously linked forward node."""
    if node in self._forward_link_nodes:
      self._forward_link_nodes.remove(node)

  @property
  def total_values(self) -> int:
    """Total number of possible values (flattened)"""
    return self._total_values

  @property
  def dim_sizes(self) -> Tuple[int, ...]:
    """Shape of the multi-dimensional space (or (N,) for 1D)"""
    return self._dim_sizes
  
  @dim_sizes.setter
  def dim_sizes(self, sizes: Sequence[int]) -> None:
    sizes = tuple(int(x) for x in sizes)
    if any(s <= 0 for s in sizes):
      raise ValueError('All dimension sizes must be positive integers')
    
    # Compute strides
    strides = [1]*len(sizes)
    for i in range(len(sizes)-2, -1, -1):
      strides[i] = sizes[i+1] * strides[i+1]
    
    total = strides[0] * sizes[0] if sizes else 1
    
    self._dim_sizes = sizes
    self._dim_strides = tuple(strides)
    self._total_values = total
      
  @property
  def value_tuple(self) -> Tuple[int, ...]:
    """Multiple-dimensional coordinates (big-endian order)"""
    if not self._dim_sizes:
      return (self._value,)
    coords = []
    remainder = self._value
    for stride in self._dim_strides:
      coord = remainder // stride
      coords.append(coord)
      remainder -=  coord*stride
    return tuple(coords)
    
  @value_tuple.setter
  def value_tuple(self, coords: Sequence[int]) -> None:
    if len(coords) != len(self._dim_sizes):
      raise ValueError(
        f"Expected {len(self._dim_sizes)} coordinates, got {len(coords)}")
    val = 0
    for i, c in enumerate(coords):
      dim_size = self._dim_sizes[i]
      if not (0 <= c < dim_size):
        raise ValueError(
          f"Coordinate {c} out of bounds for dimension {i} "
          f"(valid range: 0 <= x < {dim_size})")
      val += c * self._dim_strides[i]
    self._value = val

  def short_repr(self) -> str:
    """Short representation - useful for graph drawing / debugging"""
    if len(self._dim_sizes) > 1:
      return f"{self.name} = {self.value_tuple}"
    else:
      return f"{self.name} = {self._value}"

      
class DecisionNode(ValueDAGNode):
  """
  Specialized ValueDAGNode representing a decision node.
  
  Attributes:
  -----------
  decision_type    : str, indicates how the decision is to be made (maximize
                     expected utility, etc.) valid types: 'max','min'

  Methods:
  --------
  __init__         : class constructor
  """
    
  VALID_TYPES = frozenset({'max', 'min'})
    
  def __init__(self, name: str, /, **kwargs):
    """
    Args:
    -----------
    name              : str, required node identifier
    **kwargs : 
      - decision_type : str = 'max'
      - total_values  : Optional[int]
      - dim_sizes     : Optional[Sequence[int]]
      - value         : Optional[int]
    """
    
    decision_type = kwargs.pop('decision_type','max')
    
    if decision_type not in self.VALID_TYPES:
      raise ValueError(
        f"Invalid decision type {decision_type!r} for node {name!r}. "
        f"Valid options: {', '.join(sorted(self.VALID_TYPES))}")
    
    # Forward remaining kwargs to the parent class
    super().__init__(name=name, **kwargs)
    
    self.decision_type: str = decision_type

class ChanceNode(ValueDAGNode):
  """
  Specialized ValueDAGNode representing a chance/probability node.
  
  Models a discrete value whose probability distribution is either:
  - static (no parents) -> Static Probability Table (SPT)
  - conditional on parent values -> Conditional Probability Table (CPT)
  
  The probability table is a multi-dimensional numpy array where:
  - the first dimensions correspond to the possible values of each parent
  - the last dimension corresponds to the possible values of this node
  - each "slice" over the last dimension must sum to 1.0 (valid probability
    distribution)

  Attributes:
  -----------
  _table            : np.ndarray[float] probability distribution
                      
                        P1 x P2 x ... x PN x M of float where
                        Pj represents the valus for the jth parent and
                        M represents the value for self
                        
                        sums over the last column (M) must equal 1.0
  
  Properties:
  -----------
  table               : np.ndarray[float] - probability table (shaped determined
                        by parents + self)

  Methods:
  --------
  __init__            : class constructor
  set_random_value    : none, sample a value according to the probability
                        distribution
  get_probability     : float, return P(self.value| parents' values)
  create_random_table : none, initialize table with random probabilities
  verify_table        : none, check table shape and probability constraints
  """
  
  def __init__(self, name: str, /, **kwargs):
    """
    Args:
    -----------
    name              : str, required node identifier
    **kwargs : 
      - table             : probability table
      - init_random_value : logical to initialize value according to prob. dist.
      - total_values      : number of possible values (for 1D case)
      - dim_sizes         : tuple/list of dimension sizes (for multi-D case)
      - value             : initial value (will be validated)
    """
    table = kwargs.pop('table',None)
    init_random_value = kwargs.pop('init_random_value',False)
    
    # Forward remaining kwargs to the parent class
    super().__init__(name=name, **kwargs)
    
    self.table = table
    if init_random_value:
      self.verify_table()
      self.set_random_value()
      
  @property
  def table(self) -> np.ndarray:
    """The probability table as a numpy array (dtype=float)."""
    return self._table
  
  @table.setter
  def table(self, val: Any) -> None:
    """
    Set probability table - accepts flexible input formats.
    
    Args:
    ------------
    val : any, must be convertible to np.ndarray[float], None create empty array
    
    Performs:
    ------------
    conversion to np.ndarray[float]
    shape compatibility check along self (last) dimension only 
    normalization check (sums ~= 1.0 along last axis)
    """
    if val is None:
      self._table = np.array([], dtype=float)
      return
    
    # Convert to numpy array
    try:
      arr = np.asarray(val, dtype=float)
    except (ValueError,TypeError) as e:
      raise TypeError(
        f"Cannot convert input to probability table: {e}\n"
        f"Supported: np.ndarray, list[list[float]], etc."
      ) from e
      
    # Shape check if we know our own size
    if self._total_values > 1 and arr.size > 0:
      if arr.shape[-1] != self._total_values:
        raise ValueError(
          f"Last dimension ({arr.shape[-1]}) does not match "
          f"node's total_values ({self._total_values})"
        )
    
    # Normalization check (allow small float errors)
    if arr.size > 0:
      sums = arr.sum(axis=-1)
      if not np.allclose(sums, 1.0, atol=1e-9, rtol=1e-5):
        raise ValueError(
          "Invalid probability table: sums along last axis are not ~= 1.0"
        )
    
    self._table = arr

  def set_random_value(self) -> None:
    """
    Sample a value for this node according to the probability distribution.
    
    Parameters:
    -----------
    None
    """
    
    # Check table is usable
    if self.table.size == 0:
      raise ValueError("Cannot sample: probability table is empty")
    
    if self.table.ndim == 0:
      raise ValueError("Cannot sample: table has no dimensions")
    
    if self.table.ndim != len(self.parents) + 1:
      raise ValueError("Table shape does not match number of parents + self")
    
    parents_index = tuple([p.value for p in self.parents])
    probs = self.table[parents_index]
    
    if not np.allclose(probs.sum(), 1.0, atol=1e-9):
      raise ValueError("Probability slice does not sum to 1.0")
    
    if np.any(probs < 0):
      raise ValueError("Negative probabilities detected")
    
    self.value = int(np.random.choice(a=range(len(probs)),p=probs/probs.sum()))

  def get_probability(self) -> float:
    """
    Calculate the probability for the value of this node conditioned on the
    values of the parent nodes.
        
    Parameters:
    -----------
    None
    
    Returns:
    --------
    probability : float, in [0,1]
    """
    return self._table[tuple([p.value for p in self.parents] + [self._value])]
      
  def create_random_table(self, seed: int | None = None):

    shape = [p.total_values for p in self.parents] + [self._total_values]
    self.table = cpt_utils.create_cpt_random(shape, seed = seed)
    self.verify_table()
      
  def verify_table(self) -> None:
    """
    Verify that the probability table is valid with respect to:
    - shape compatibility with parent nodes and self
    - non-negative probabilities
    - sums to 1.0 (within floating-point tolerance) along the last axis
    
    Raises:
      ValueError: if any check fails (with detailed message)
      RuntimeError: if table is not set or has invalid type
        
    Parameters:
    -----------
    None
    """
    
    # Basic existence and type checks
    if not hasattr(self,'_table') or self._table is None:
      raise RuntimeError("Probability table is not set (self._table is None)")
    
    if not isinstance(self._table, np.ndarray):
      raise RuntimeError(
        f"Table must be a numpy ndarray, got {type(self._table).__name__}"
      )
    
    if self._table.size == 0:
      # Allow empty table only if no parents and no values defined yet
      if not self.parents and self._total_values <= 1:
        return
      raise ValueError("Table is empty but node has defined dimensions")
      
    expected_shape = [p.total_values for p in self.parents] + \
                     [self.total_values]
    
    cpt_utils.verify_cpt(expected_shape, self._table)
        
  def short_repr(self) -> str:
    """Short representation - useful for debugging"""
    if len(self._dim_sizes) > 1:
      return f"{self.name} = {self.value_tuple}; table = {self._table}"
    else:
      return f"{self.name} = {self._value}; table = {self._table}"
      
      
class StateNode(ChanceNode):
  """
  A visible/observed state variable whoe value is known or directly 
  controllable.

  Recommended for
  - controlled or user-set states (position, heading, mode switch, etc.)
  
  DecisionNodes are aware of StateNode values (if they appear after the state
  in temporal order).
  """

  def __init__(self, name: str, /, **kwargs: Any):
    """
    Same parameters as ChanceNode.
    """
    super().__init__(name, **kwargs)

  def short_repr(self) -> str:
    """Include type indicator for easier debugging/visualization."""
    base = super().short_repr()
    return f"{base} [STATE]"
    
    
class EvidenceNode(ChanceNode):
  """
  A node representing directly observed evidence or measurement data.

  Recommended for:
  - sensor readings, test results, user reports, or any external observation

  DecisionNodes are aware of EvidenceNode values (if they appear after the 
  evidence in temporal order).
  """

  def __init__(self, name: str, /, **kwargs: Any):
    """
    Same parameters as ChanceNode.
    """
    super().__init__(name, **kwargs)

  def short_repr(self) -> str:
     """Include type indicator for easier debugging/visualization."""
     base = super().short_repr()
     return f"{base} [EVIDENCE]"
    
    
class HiddenStateNode(ChanceNode):
  """
  A latent / unobservable state variable in the model.

  The value cannot be directly seen, but can be inferred from evidence and other
  nodes (e.g. flu hidden state can be inferred from thermometer evidence).

  DecisionNodes are **not** aware of HiddenStateNode values, regardless of
  temporal order.
  """

  def __init__(self, name: str, /, **kwargs: Any):
    """
    Same parameters as ChanceNode.
    """
    super().__init__(name, **kwargs)

  def short_repr(self) -> str:
    """Include type indicator for easier debugging/visualization."""
    base = super().short_repr()
    return f"{base} [HIDDEN]"


class UtilityNode(DAGNode):
  """
  Specialized DAGNode representing a utility node.
  
  This node has no discrete value. The utility of this node is obtained by
  indexing the parent values into the table (no own dimension).
    
  Attributes:
  -----------
  _table            : np.darray[float] utility values, shape determined by
                      parents
  
                      P1 x P2 x ... x PN of float where
                      Pj represents the value of the jth parent

  Properties:
  -----------
  table               : utility array (read/write with validation)

  Methods:
  --------
  __init__            : class constructor
  get_utility         : float, return P(parents' values)
  create_random_table : none, initialize table with random utilities
  verify_table        : none, check table shape and probability constraints
  """
  
  def __init__(self, name: str, /, **kwargs):
    """
    Args:
    -----------
    name              : str, required node identifier
    **kwargs : 
      - table         : probability table
    """
    table = kwargs.pop('table',None)
    super().__init__(name=name, **kwargs) # fwd remaining to super class
    self.table = table
      
  @property
  def table(self) -> np.ndarray:
    """The utility table as a numpy array (dtype=float)."""
    return self._table
  
  @table.setter
  def table(self, val: Any) -> None:
    """
    Set utility table - accepts flexible input formats.
    
    Args:
    ------------
    val : any, must be convertible to np.ndarray[float], None create empty array
    
    Performs:
    ------------
    conversion to np.ndarray[float]
    """
    if val is None:
      self._table = np.array([], dtype=float)
      return
    
    # Convert to numpy array
    try:
      arr = np.asarray(val, dtype=float)
    except (ValueError,TypeError) as e:
      raise TypeError(
        f"Cannot convert input to probability table: {e}\n"
        f"Supported: np.ndarray, list[list[float]], etc."
      ) from e
      
    self._table = arr

  def get_utility(self) -> float:
    """
    Calculate the utility for this node based on the parent nodes.
    
    Parameters:
    -----------
    None
    
    Returns:
    --------
    utility : float, in [0,1]
    """
    return self._table[tuple([p.value for p in self.parents])]
      
  def create_random_table(self, seed=None):
    
    if seed is not None:
      np.random.seed(seed)
    
    shape = [p.total_values for p in self.parents]
    arr = np.random.rand(*shape)
    self.table = arr

    self.verify_table()
      
  def verify_table(self) -> None:
    """
    Verify that the utility table is valid with respect to:
    - shape compatibility with parent nodes and self
    
    Raises:
      ValueError: if any check fails (with detailed message)
      RuntimeError: if table is not set or has invalid type
        
    Parameters:
    -----------
    None
    """
    
    # Basic existence and type checks
    if not hasattr(self,'_table') or self._table is None:
      raise RuntimeError("Utility table is not set (self._table is None)")
    
    if not isinstance(self._table, np.ndarray):
      raise RuntimeError(
        f"Table must be a numpy ndarray, got {type(self._table).__name__}"
      )
      
    if self._table.dtype.kind not in 'fd':
      raise ValueError("Table must have floating-point dtype (float or double)")
    
    if self._table.size == 0:
      # Allow empty table only if no parents and no values defined yet
      if not self.parents:
        return
      raise ValueError("Table is empty but node has defined dimensions")
      
    # Check number of dimensions
    expected_ndim = len(self.parents)
    if self._table.ndim != expected_ndim:
      raise ValueError(
        f"Table has {self._table.ndim} dimensions, "
        f"but expected {expected_ndim} (parents)"
      )
      
    # Check shape matches parent total_values
    expected_shape = [p.total_values for p in self.parents]
    if self._table.shape != tuple(expected_shape):
      raise ValueError(
        f"Table shape {self._table.shape} does not match expected shape "
        f"{tuple(expected_shape)} (parents' total_values)"
      )
        
  def short_repr(self) -> str:
    """Short representation - useful for debugging"""
    return f"{self.name}; table = {self._table}"


def main():

  pass

if __name__ == '__main__':
  sys.exit(main())
