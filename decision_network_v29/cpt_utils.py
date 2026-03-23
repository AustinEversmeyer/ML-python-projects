#cpt_utils.py
"""
Utilities for working with Conditional Probability Tables (CPTs).

A CPT is a numpy array of shape (d1, d2, ..., dn, m) where:
- d1..dn are the dimensions of the conditioning variables (parents)
- m is the number of outcomes for the conditioned variable (child)

Each slice over the last dimension must sum to 1.0.
"""

from __future__ import annotations
import numpy as np
from math import prod
from typing import Tuple, Sequence

def verify_cpt(shape: Sequence[int], table: np.ndarray) -> None:
  """
  Verify that a CPT array is valid:

  - Has the correct shape
  - dtype is floating-point
  - All values are finite and non-negative
  - Each slice over the last dimension sums to 1.0 (within tolerance)

  Raises:
    ValueError: if any check fails
  """
  expected_shape = tuple(shape)
  if table.shape != expected_shape:
    raise ValueError(
      f"Table shape {table.shape} does not match "
      f"expected shape {expected_shape}"
    )

  if not np.issubdtype(table.dtype, np.floating):
    raise ValueError("Table must have floating-point dtype")

  if not np.all(np.isfinite(table)):
    raise ValueError("Table contains NaN or infinite values")

  if np.any(table < 0):
    raise ValueError("Table contains negative probabilities")

  # Sum over last axis, allow small floating-point error
  sums = table.sum(axis=-1)
  if not np.allclose(sums, 1.0, atol=1e-10, rtol=1e-8):
    raise ValueError(
      "Not all slices sum to 1.0 (within tolerance). "
      f"Min sum = {sums.min():.12f}, Max sum = {sums.max():.12f}"
    )


def create_cpt_random(
  shape: Sequence[int], 
  seed: int | None = None
) -> np.ndarray:
  """
  Create a random valid CPT with the given shape.

  The last dimension is the child variable's outcomes.
  Each slice over the last dimension sums to 1.0.

  Args:
    shape: full shape, e.g. (2, 3, 4) means 2×3 parents, 4 child outcomes

  Returns:
    np.ndarray of shape `shape` with dtype float64, valid probabilities
  """
  if not shape or shape[-1] <= 0:
    raise ValueError("Shape must be non-empty and last dimension > 0")

  # Set seed if provided
  if seed is not None:
    np.random.seed(seed)

  # Generate raw random values
  arr = np.random.rand(*shape)
  arr += 0.02
  
  # Normalize over the last axis
  arr /= arr.sum(axis=-1, keepdims=True)


  # Fix any floating-point issues near zero
  arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
  arr = np.clip(arr, 0.0, 1.0)

  # Re-normalize
  arr /= arr.sum(axis=-1, keepdims=True)
  
  verify_cpt(shape, arr)

  return arr
  

def cpt_adjust_zeros(table: np.ndarray) -> np.ndarray:
  """
  Adjust a CPT by only modifying zero entries so that each slice sums to 1.0.

  - Non-zero values stay unchanged.
  - Zeros in each slice are equally increased to make the slice sum to 1.0.
  - Calls verify_cpt before and after adjustment.

  Returns:
    Adjusted table (new copy, original unchanged)
  """

  arr = table.copy()  # work on copy

  # For each slice over the last dimension
  for idx in np.ndindex(arr.shape[:-1]):
    slice_ = arr[idx]
    zero_mask = slice_ == 0
    num_zeros = zero_mask.sum()

    if num_zeros == 0:
      continue

    # Current sum of non-zeros
    current_sum = slice_.sum()
    deficit = 1.0 - current_sum

    if deficit < 0:
      raise ValueError(
        f"Slice {idx} sums to {current_sum:.6f} > 1.0 — cannot adjust zeros"
      )

    # Distribute deficit equally to zeros
    arr[idx][zero_mask] += deficit / num_zeros

  verify_cpt(table.shape, arr)
  return arr


def cpt_adjust(table: np.ndarray) -> np.ndarray:
  """
  Adjust a CPT so every slice over the last dimension sums exactly to 1.0.

  - All values are scaled equally in each slice.
  - Calls verify_cpt before and after adjustment.

  Returns:
    Adjusted table (new copy, original unchanged)
  """
  arr = table.copy()  # work on copy

  # For each slice over the last dimension
  for idx in np.ndindex(arr.shape[:-1]):
    slice_ = arr[idx]

    # Current sum of non-zeros
    current_sum = slice_.sum()
    deficit = 1.0 - current_sum

    if deficit < 0:
      raise ValueError(
        f"Slice {idx} sums to {current_sum:.6f} > 1.0 — cannot adjust"
      )

    # Distribute deficit equally to zeros
    arr[idx] += deficit / arr.shape[-1]

  verify_cpt(table.shape, arr)
  return arr