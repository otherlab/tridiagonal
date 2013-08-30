#!/usr/bin/env python

from __future__ import division
from numpy import *

def tridiagonal_dot(A,x):
  '''Multiply a symmetric diagonal matrix times a vector.  For some reason, this is missing from scipy.'''
  assert A.shape==(len(x),2) 
  y = A[:,0]*x
  y[:-1] += A[:-1,1]*x[1:]
  y[1:]  += A[:-1,1]*x[:-1]
  return y

def tridiagonal_qp_error(A,b,lo,hi,x,tol=1e-10):
  '''Given x supposedly minimizing E = 1/2 x'Ax - b'x subject to lo <= x <= hi, compute the error in the KKT conditions.

  Parameters
  ----------
  A,b,lo,hi : As for tridiagonal_qp
  x : Candidate solution 

  Returns
  -------
  bound : how far each component of x exceeds the bounds
  kkt : KKT error for each component of x.  e is the energy gradient corrected for active constraints.
  '''

  # Check input consistency
  assert len(b)==len(lo)==len(hi)==len(x)
  n = len(b)
  assert A.shape==(n,2)
  if not n:
    return zeros(0)

  # Compute errors
  bound = maximum(0,maximum(lo-x,x-hi))
  Ex = tridiagonal_dot(A,x)-b
  kkt = (  Ex
         * (1-((abs(x-lo)<tol)&(Ex>0)))  # If x==lo and Ex>0, the lo constraint is active
         * (1-((abs(x-hi)<tol)&(Ex<0)))) # If x==hi and Ex<0, the hi constraint is active
  return bound,kkt
