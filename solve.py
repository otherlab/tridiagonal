#!/usr/bin/env python

from __future__ import division
from numpy import *
from collections import deque
from scipy import linalg

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

def tridiagonal_qp(A,b,lo,hi,mode='fast'):
  '''Find x minimizing E = 1/2 x'Ax - b'x subject to lo <= x <= hi, where A is tridiagonal and symmetric positive definite.
  The algorithm is a linear time active set method analogous to convex hull of sorted points.

  Parameters
  ----------
  A : The SPD tridiagonal matrix A given as an (n,2) array.  A[0] is the diagonal and A[1,:-1] is the offdiagonal
  b : Linear term of the objective
  lo,hi : Bounds on x, which may be infinite
  mode : 'fast' or 'slow', for unstable linear time or stable quadratic time (which can be straightforwardly optimized to O(n log n))
  '''

  # Check input consistency
  assert len(b)==len(lo)==len(hi)
  n = len(b)
  assert A.shape==(n,2)
  assert all(lo<=hi)
  assert all(A[:,0]>=0), 'The diagonal of A must be strictly positive'
  if n==0:
    return b
  assert mode in ('fast','slow','debug')

  '''
  # Compute cholesky factorization A = LL'.
  L = scipy.linalg.cholesky_banded(A.T,lower=True).T
  L[-1,1] = 0
  assert all(L[:,0] > 0)
  c = scipy.linalg.cho_solve_banded((L.T,True),b)
  # Now E = 1/2 | L^T x + c |^2 + const
  '''

  '''
  # Adjust signs so that L[:,1] <= 0
  scale = 1-2*hstack([0,logical_xor.accumulate(L[:-1,1]>0)])
  L[:-1,1] *= scale[1:]*scale[:-1]
  c *= scale
  lo *= scale
  hi *= scale
  '''

  # Offset x so that b = 0
  base = linalg.solveh_banded(A.T,b,lower=True)
  lo = lo-base
  hi = hi-base

  # Adjust signs so that offdiagonal terms are negative to ensure monotonicity
  scale = 1-2*hstack([0,logical_xor.accumulate(A[:-1,1]>0)])
  A = A.copy()
  A[:-1,1] *= scale[1:]*scale[:-1]
  assert all(A[:-1,1]<=0)
  lo,hi = sort(scale*[lo,hi],axis=0)

  # Write every variable as a linear combination of source RHSs at the two ends
  #  x = L b[0] + R b[-1]
  def slow_middle(i,xi,k,xk,j):
    '''Compute xj given xi and xk in O(k-i) time using full tridiagonal solves'''
    assert -1<=i<=j<=k<=n
    assert 0<=j<n
    if i==j:
      return xi
    if j==k:
      return xk
    jb = zeros(k-i-1)
    if 0<=i:
      jb[0] -= A[i,1]*xi
    if k<n:
      jb[-1] -= A[k-1,1]*xk
    return linalg.solveh_banded(A[i+1:k].T,jb,lower=True)[j-i-1]

  if mode in ('fast','debug'):
    L = linalg.solveh_banded(A.T,hstack([1,zeros(n-1)]),lower=True)
    R = linalg.solveh_banded(A.T,hstack([zeros(n-1),1]),lower=True)
    LR = hstack([L[:,None],R[:,None]])

    def fast_middle(i,xi,k,xk,j):
      '''Compute xj given xi and xk unstably in O(1) time by cancelling linear combination weights'''
      assert -1<=i<=j<=k<=n
      assert 0<=j<n
      if 0<=i:
        if k<n:
          # xi = Li a + Ri b
          # xk = Lk a + Rk b
          # xj = Lj a + Rj b
          return dot(LR[j],linalg.solve(vstack([LR[i],LR[k]]),(xi,xk)))
        else: # k==n
          assert xk is None
          # b = 0
          # xi = Li a + Ri b = Li a
          # xj = Lj a + Rj b = Lj a
          return xi*L[j]/L[i]
      else: # -1==i
        if k<n:
          assert xi is None
          # a = 0
          # xk = Lk a + Rk b = Rk b
          # xj = Lj a + Rj b = Rj b
          return xk*R[j]/R[k]
        else: # k==n
          # If there are no outer constraints, x = 0 since we've shifted to b = 0 above.
          return 0

  if mode=='slow':
    middle = slow_middle
  elif mode=='fast':
    middle = fast_middle
  elif mode=='debug':
    def middle(*args):
      fast = fast_middle(*args)
      slow = slow_middle(*args)
      print 'fast %g, slow %g'%(fast,slow)
      assert abs(fast-slow)<1e-5
      return fast

  # On to the core algorithm.  Our internal state consists of
  #
  # 1. A list of known x values common to both lower and upper paths.
  # 2. An optimal lower path ending at x[k] = lo[k], represented as a deque of indices i where x[i] = lo[i].
  # 3. An optimal upper path analogous to lower but for x[k] = hi[k] and x[i] = hi[i].
  #
  # Each iteration, we expand the lower and upper paths by one entry (one constraint), then snap off unnecessary intermediate constraints until optimality is restored.
  prefix = [None]
  lower = deque()
  upper = deque()
  def extend(k,sign):
    # Extend one of the paths, then erode down that path and possibly up the other until we've restored convexity
    assert abs(sign)==1
    if sign>0:
      c0,c1 = lo,hi
      path0,path1 = lower,upper 
    else:
      c0,c1 = hi,lo
      path0,path1 = upper,lower
    # Reduce path0 as far as necessary
    xk = c0[k] if k<n else None
    while path0:
      j = path0[-1]
      try:
        i = path0[-2]
        xi = c0[i]
      except IndexError:
        i = len(prefix)-2
        xi = prefix[-1]
      xj = middle(i,xi,k,xk,j)
      if sign*xj <= sign*c0[j]: # If the c0[j] constraint is necessary, stop eroding
        break
      else: # Otherwise, discard the c0[j] constraint and keep reducing
        path0.pop()
    else:
      # We've exhausted path0, so start eroding path1 from the start.
      while path1:
        i = len(prefix)-2
        xi = prefix[-1]
        j = path1[0]
        if j==k:
          break
        xj = middle(i,xi,k,xk,j)
        if sign*xj <= sign*c1[j]: # If the c1[j] constraint doesn't conflict, stop eroding
          break
        else: # If the c1[j] constraint does conflict, the shared path must include x[j] = c1[j].  Freeze the path portion from xi to c1[j].
          for ij in xrange(i+1,j):
            prefix.append(middle(i,xi,j,c1[j],ij))
          prefix.append(c1[j])
          path1.popleft()
    path0.append(k)
  for k in xrange(0,n+1):
    # Extend lower and upper paths, reducing until we've restored convexity
    extend(k,+1) # Lower
    extend(k,-1) # Upper
  if len(prefix)<=n:
    i = len(prefix)-2
    xi = prefix[-1]
    for j in xrange(i+1,n):
      prefix.append(middle(i,xi,n,None,j))

  # Extract the result and undo our simplifying transforms
  return base+scale*prefix[1:]

'''
    upper.append(entry)

  Consider x0,x1,x2.  We can either have an energy

    E = 1/2 (L00 x0 + L10 x1 + c0)^2 + 1/2 (L11 x1 + L21 x2 + c1)^2

  or stick with the tridiagonal version

    a10 x0 + a11 x1 + a12 x2 = b1

  Naturally the tridiagonal version is easiest in terms of generating intermediate values, since

    x1 = (b1 - a10 x0 - a12 x2) / a11

  What about intermediate variable elimination?  Each variable is touched by three equations, which reduces to two matrix coefficients by symmetry.
  Write di = aii, oi = d[i,i+1]

    x0,x1,x2,x3,x4

    o01 x0 + d1 x1 + o12 x2 = b1
    o12 x1 + d2 x2 + o23 x3 = b2
    o23 x2 + d3 x3 + o34 x4 = b3

  Eliminating x2, we arrive at

    x0,x1,x3,x4

    x2 = (b2 - o12 x1 - o23 x3) / d2

    o01 x0 + d1 x1 + o12 (b2 - o12 x1 - o23 x3) / d2 = b1
    o01 x0 + (d1 - o12^2/d2) x1 - o23 o12/d2 x3 = b1 - b2 o12/d2

    o23 (b2 - o12 x1 - o23 x3) / d2 + d3 x3 + o34 x4 = b3
    -o12 o23/d2 x1 - (d3 - o23 o23/d2) x3 + o34 x4 = b3 - b2 o23/d2

    o01,o34 unchanged 
    d1 -> d1 - o12 o12/d2
    b1 -> b1 - b2  o12/d2
    d3 -> d3 - o23 o23/d2
    b3 -> b3 - b2  o23/d2
    o13 = -o23 o12/d2

  What happens in the energy case?

    E = 1/2 (L00 x0 + L10 x1 + c0)^2 + 1/2 (L11 x1 + L21 x2 + c1)^2
    E_x1 = (L00 x0 + L10 x1 + c0) L10 + (L11 x1 + L21 x2 + c1) L11 = 0
    x1 = - (c0 L10 + c1 L21 + L00 L10 x0 + L21 L11 x2) / (L10^2 + L11^2) = a0 x0 + a2 x2 + d
    E = 

  What about the tridiagonal case where b = 0?  We have

    o01,o34 unchanged
    d1 -> d1 - o12 o12/d2
    d3 -> d3 - o23 o23/d2
    o13 = -o23 o12/d2

  Say initially all diagonal entries are 1 and offdiagonals are -o.  Cyclic elimination produces

     d_ = 1 - o^2
     o_ = -o^2
   
     d__ = d_ - o^4 / (1 - o^2) = 1 - o^2 - o^4/(1-o^2) = (1-2o^2+o^4 - o^4)/(1-o^2) = (1-2o^2)/(1-o^2) ~ (1-2o^2)(1+o^2+o^4
     o__ = -o^4 / (1-o^2)
'''
