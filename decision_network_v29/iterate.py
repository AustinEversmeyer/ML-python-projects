from __future__ import annotations
from math import prod
import numpy as np
from typing import Tuple, Union, List, Sequence, Optional, Iterator


class MultiDimIterator(Iterator[Tuple[int, ...]]):
  """
  Iterator over all valid multi-dimensional indices for a given shape.
  
  Yields tuples of indices in C-order (last dimnension varies fastest).
  
  Examples:
    MultiDimIterator((2,3)) -> (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
  """
  
  def __init__(self, shape: Union[List[int], Tuple[int, ...]]):
    """
    Args:
      shape: tuple of positive integers (or empty)
    Raises:
      ValueError: if shape contains non-positive integers
    """
    if not isinstance(shape, tuple):
      raise TypeError("shape must be a tuple")
    
    if len(shape) == 0 or any(s <= 0 for s in shape):
      raise ValueError("Shape must be a nonempty tuple of positive integers.")
    
    self._shape: Tuple[int, ...] = tuple(shape)
    self._ndim: int = len(shape)
    self._indices: List[int] = [0] * self._ndim
    self._done: bool = False
    
  def reset(self) -> None:
    """Reset the iterator to the beginning (all indices = 0, not done)."""
    self._indices = [0]*self._ndim
    self._done = False

  def __iter__(self) -> Iterator[Tuple[int, ...]]:
    return self

  def __next__(self) -> Tuple[int, ...]:
    if self._done:
      raise StopIteration
    
    # Yield current position
    current = tuple(self._indices)
    
    # Increment the indices (last dimension fastest)
    i = self._ndim - 1
    while i >= 0:
      self._indices[i] += 1
      if self._indices[i] < self._shape[i]:
        break
      self._indices[i] = 0
      i -= 1
    
    # If we carried over past the first dimension, we're done
    if i < 0:
      self._done = True
    
    return current
  
  def __len__(self) -> int:
    """
    Return the total number of elements in the multi-dimensional space.
    Returns 0 if any dimension is zero.
    """
    if self._done:
      return 0
    prod = 1
    for s in self._shape:
      prod *= s
    return prod
    
  def __repr__(self) -> str:
    return f"{self.__class__.__name__}(shape={self._shape})"
    

class MaskedMultiDimIterator(Iterator[Union[Tuple[int, ...], int]]):
  """
  Iterator over masked multi-dimensional indices.

  Only yields positions where mask is True.
  Stores only the flat indices of True values.

  Args:
    _shape          : dimension sizes (tuple)
    _ndim           : number of dimensions (int)
    _total          : number of iterations un-masked, product of shape (int)
    _inner_size     : size of innermost/rightmost dimension (int)
    _total_prefix:    size of inner dimensions, product of shape[:-1] (int)
    _mask           : length total boolean, T = use, F = skip
    _positions      : list of positions of true masks
    _pos_idx        : index into _positions
    _next_type      : one of 'flat', 'fpi', 'coord' indicating to return the
                      flat index, a tuple of (flat,prefix,inner), or 
                      a coordinate tuple
    _flat_idx           : flat index for all dimensions
    _inner_idx          : index for the last dimension (single dim so flat)
    _flat_prefix_idx    : flat index for all dimensions up to the last
    
  Properties:
    coord_idx           : current coordinate tuple index
    flat_idx            : current flat index
    flat_prefix_idx     : current flat index of the prefix dimensions
    inner_idx           : current index of the rightmost/last dimension

  Yields:
    Tuple[int, ...] if yield_coords = True
    int (flat index) if yield_coords = False (faster)
  """

  def __init__(
    self,
    shape: Union[List[int], Tuple[int, ...]],
    mask: Union[List[bool], Tuple[bool, ...]] = None,
    next_type: str = 'flat',
  ):
    """
    Args:
      shape: dimension sizes (tuple or list of int)
      mask: flat boolean sequence, length == prod(shape), if None, all positions
            are included
      yield_coords: if True, yield (i,j,k,...), else yield flat index
    """
    
    shape = tuple(shape)
    if not shape or any(s <= 0 for s in shape):
      raise ValueError("shape must be tuple/list of positive integers")
    
    if not next_type in ('flat','fpi','coord'):
      raise ValueError("next type must be one of flat, fpi, or coord")
    
    self._shape: Tuple[int, ...] = shape
    self._ndim: int = len(shape)
    self._total: int = prod(self._shape)
    self._next_type: str = next_type
    
    self._inner_size: int = shape[-1] if self._ndim >= 1 else 1
    self._total_prefix = self._total // self._inner_size

    if mask is None:
      self._mask = [True,]*self._total
    else:
      self._mask = mask

    if len(self._mask) != self._total:
      raise ValueError(f"mask length must be {self._total}, got {len(mask)}")

    # Only store positions where mask is True
    self._positions: List[int] = [i for i, v in enumerate(self._mask) if v]
    
    self.reset()

  def __iter__(self) -> Iterator[Union[Tuple[int, ...], 
                                 Tuple[int, int, int], int]]:
    return self

  def __next__(self) -> Union[Tuple[int, ...], Tuple[int, int, int], int]:
    if self._pos_idx >= len(self._positions):
      raise StopIteration

    flat = self._positions[self._pos_idx]
    self._pos_idx += 1

    # Update tracking indices (always)
    self._flat_idx = flat
    self._inner_idx = flat % self._inner_size
    self._flat_prefix_idx = flat // self._inner_size

    if self._next_type == 'flat':
      return flat
    elif self._next_type == 'fpi':
      return (flat, self._flat_prefix_idx, self._inner_idx)
    else:
      return self._flat_to_coord(flat)

  def _flat_to_coord(self, flat: int) -> Tuple[int, ...]:
    """Convert flat index, n-dimensional coordinate (slow part)"""
    coord = []
    x = flat
    for s in reversed(self._shape):
      coord.append(x % s)
      x //= s
    coord.reverse()
    return tuple(coord)
    
  @property
  def coord_idx(self) -> Optional[Tuple[int, ...]]:
    """
    Compute / return current coordinate even when yield_coords=False.
    Returns None before first __next__() call.
    Computres lazily if needed.
    """
    return tuple(self._flat_to_coord(self._flat_idx))

  @property
  def flat_idx(self) -> int:
    return self._flat_idx

  @property
  def flat_prefix_idx(self) -> int:
    """Flat index over prefix dimensions (shape[:-1])."""
    return self._flat_prefix_idx

  @property
  def inner_idx(self) -> int:
    """Index in the innermost (last/right) dimnension"""
    return self._inner_idx
    
  def reset(self) -> None:
    """Reset iterator to beginning"""
    self._pos_idx = 0
    self._flat_idx = -1
    self._flat_prefix_idx = -1
    self._inner_idx = -1

  def __len__(self) -> int:
    if self._pos_idx >= len(self._positions):
      return 0
    return len(self._positions)
    
  def __repr__(self) -> str:
    return (f"{self.__class__.__name__}("
            f"shape={self._shape}, "
            f"masked={len(self._positions)}/{self._total}, "
            f"next_type={self._next_type})")
    
class IteratorReducer:
  """
  Reduces values for a multiple dimensional iterator according to the
  operations: sum=add, min, max, none. Achieves this by constructing a sequence
  of iterators.
  
  For shape = (2,3,2):
  iterator[0] is MaskedMultiDimIterator for shape = (2,)
  iterator[1] is MaskedMultiDimIterator for shape = (2,3)
  iterator[2] is MaskedMultiDimIterator for shape = (2,3,2)
  
  Reductions are applied over sets consisting of a single prefix index
  e.g.
  shape = (2,3,2), ops = (sum,max,sum)
  values = [1,4,2,2,3,4,5,1,3,2,1,7]
    
    [0,0,0] 1 sum --> [0,0] 5  --
    [0,0,1] 4                    \
    [0,1,0] 2 sum --> [0,1] 4  ---> max [0,] 7 \
    [0,1,1] 2                    /              \
    [0,2,0] 3 sum --> [0,2] 7  --                \
    [0,2,1] 4                                     -> sum 15
    [1,0,0] 5 sum --> [1,0] 6  --                /
    [1,0,1] 1                    \              /
    [1,1,0] 3 sum --> [1,1] 5  ---> max [1,] 8 /
    [1,1,1] 2                    /
    [1,2,0] 1 sum --> [1,2] 8  --
    [1,2,1] 7
    
  with a mask of [True, True, True, True, True, False,
                  True, True, True, True, True, False]
  
    [0,0,0] 1 sum --> [0,0] 5  --
    [0,0,1] 4                    \
    [0,1,0] 2 sum --> [0,1] 4  ---> max [0,] 5 \
    [0,1,1] 2                    /              \
    [0,2,0] 3 sum --> [0,2] 3  --                \
    [0,2,1] 0                                     -> sum 11
    [1,0,0] 5 sum --> [1,0] 6  --                /
    [1,0,1] 1                    \              /
    [1,1,0] 3 sum --> [1,1] 5  ---> max [1,] 6 /
    [1,1,1] 2                    /
    [1,2,0] 1 sum --> [1,2] 1  --
    [1,2,1] 0
    
  If the outermost dimensions have operation none, they are not reduced.
  e.g.
  (a) shape = (2,3,2), ops = (none,max,sum)
      values = [1,4,2,2,3,4,5,1,3,2,1,7]
      returns [7,8]
  (b) shape = (2,3,2), ops = (none,none,sum)
      values = [1,4,2,2,3,4,5,1,3,2,1,7]
      returns [5,4,7,6,5,8]
      
  Args:
    _groups                 : tuple of (dim size(int), reduction operation(str))
    _sizes                  : tuple of dim size (int)
    _ops                    : tuple of reduction operation (str)
    _first_decision_dim     : index of leftmost dimension that has a decision
    _total                  : product of dimension sizes
    _values                 : flat np.array of values over the dimensions
    _masks                  : masks for each prefix shape
    _iterators              : MaskedMultiDimIterator for each prefix shape
    _buffera                : buffer for reduced values
    _bufferb                : buffer for reduced values
    _argbuffer              : buffer for decisions

  Multi-dimensional reduction with different ops per dimension.
  Uses existence masks to sparsify iteration at each level.
  """

  def __init__(self,
    groups: Sequence[Tuple[int, str]],
    values: Sequence[float] | None = None,
    mask_flat: Sequence[bool] | None = None,
    ):
                   
    if not groups:
      raise ValueError("groups cannot be empty")

    self._groups = tuple(groups)
    self._sizes = tuple([s for s, _ in groups])
    self._ops = [op.lower() for _, op in groups]

    # Validate 'none' only as prefix
    none_prefix_len = 0
    for op in self._ops:
      if op == 'none':
        none_prefix_len += 1
      else:
        break
    if any(op == 'none' for op in self._ops[none_prefix_len:]):
      raise ValueError("'none' only allowed as consecutive prefix")
      
    #self._first_reducing_dim: int = none_prefix_len
    self._first_decision_dim = \
       next((i for i, op in enumerate(self._ops) if op in ('min', 'max')), None)
    self._total: int = prod(self._sizes)
    
    if values == None:
      values = [0.0,]*self._total
    if len(values) != self._total:
      raise ValueError( f"values ({len(values)}) != #True ({len(self._total)})")
    self._values: np.ndarray = np.asarray(values, dtype=float)

    if mask_flat is None:
      mask_flat = [True,]*self._total
    if len(mask_flat) != self._total:
      raise ValueError(f"mask length {len(mask_flat)} != {self._total}")

    # Precompute masked positions (CHECK THIS MAYBE DONT NEED)
    self._positions_flat = [i for i, v in enumerate(mask_flat) if v]

    # Build reduced existence masks (0=outer to -1=inner)
    self._masks: List[List[bool]] = \
                                  self._build_reduced_existence_masks(mask_flat)

    # Buffers sized to largest reduced mask (this is 2nd to righmost dimension)
    max_buffer_size = len(self._masks[-2]) if len(self._masks) >= 2 else 0
    self._buffera = np.empty(max_buffer_size, dtype=float) # reduced value
    self._bufferb = np.empty(max_buffer_size, dtype=float) # reduced value
    self._argbuffer = np.full(max_buffer_size,-1,dtype=np.int32) # decision

    # Build masked iterators for each reducing prefix (0 = outer, -1 = inner)
    self._iterators = []
    prefix_shape = []
    for i, (size, op) in enumerate(self._groups):
      prefix_shape.append(size)
      if size <= 1 or op == 'none':
        self._iterators.append(None)
      else:
        it = MaskedMultiDimIterator(prefix_shape, self._masks[i])
        self._iterators.append(it)

  def _build_reduced_existence_masks(self,masks_flat):
    """Build existence masks for every level; return reversed (outer first)"""
    masks = []
    current = masks_flat

    for size in reversed(self._sizes):
      masks.append(current)
      if size == 1:
        next_mask = current[:]
      else:
        next_mask = []
        for i in range(0, len(current), size):
          chunk = current[i:i + size]
          next_mask.append(any(chunk))
      current = next_mask

    return masks[::-1] # 0 = outer, -1 = inner

  def reduce(self, new_values: Optional[Sequence[float]] = None):
    """Run full reduction chain (inner to outer)"""
    if new_values is not None:
      if len(new_values) != len(self._values):
        raise ValueError("new_values length mismatch")
      self._values = np.asarray(new_values, dtype=float)

    current = self._values.copy()
    next_buf = self._buffera
    
    arg_buf = self._argbuffer
    arg_len = -1

    for i in range(len(self._groups) - 1, -1, -1):  # inner to outer
      size, op = self._groups[i]
      it = self._iterators[i]

      if size <= 1 or op == 'none' or it is None:
        continue

      it.reset()
      if i == 0:
        next_len = 1
      else:
        next_len = len(self._masks[i-1])
      
      is_decision_step = \
                  (i == self._first_decision_dim) and (op in ('min', 'max'))
      
      if is_decision_step:
          
        arg_buf[:next_len] = -1
        arg_len = next_len

        if op == 'max':
          next_buf[:next_len] = -np.inf
          for _ in it:
            pref = it.flat_prefix_idx
            val = current[it.flat_idx]
            if val > next_buf[pref]:
              next_buf[pref] = val
              arg_buf[pref] = it.inner_idx

        elif op == 'min':
          next_buf[:next_len] = np.inf
          for _ in it:
            pref = it.flat_prefix_idx
            val = current[it.flat_idx]
            if val < next_buf[pref]:
              next_buf[pref] = val
              arg_buf[pref] = it.inner_idx

        else:
          raise ValueError(f"Unsupported op: {op}")
      
      else:

        if op in ('sum', 'add'):
          next_buf[:next_len] = 0.0
          for _ in it:
            next_buf[it.flat_prefix_idx] += current[it.flat_idx]

        elif op == 'max':
          next_buf[:next_len] = -np.inf
          for _ in it:
            pref = it.flat_prefix_idx
            next_buf[pref] = max(next_buf[pref], current[it.flat_idx])

        elif op == 'min':
          next_buf[:next_len] = np.inf
          for _ in it:
            pref = it.flat_prefix_idx
            next_buf[pref] = min(next_buf[pref], current[it.flat_idx])

        else:
          raise ValueError(f"Unsupported op: {op}")

      current = next_buf[:next_len]
      next_buf = self._bufferb if next_buf is self._buffera else self._buffera
    
    arg_result = None
    if self._first_decision_dim is not None:
      outermost_op = self._ops[self._first_decision_dim]
      if outermost_op in ('min', 'max'):
        arg_result = arg_buf[:arg_len].astype(int)

    return current.copy(),arg_result
    
  def __len__(self) -> int:
    return len(self._masks[-1])
    
  def __repr__(self) -> str:
    return (f"{self.__class__.__name__}("
            f"shape={self._sizes}, "
            f"masked={sum(self._masks[-1])}/{len(self._masks[-1])}, "
            f"reductions={' '.join(self._ops)})")