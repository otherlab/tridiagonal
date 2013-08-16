#!/usr/bin/env python

from __future__ import division
from numpy import *
from collections import deque
from scipy import linalg

# Our representation for the inverse of a tridiagonal matrix has to deal with extremely large exponents.
# Therefore, we use the representation x = (y,n) = y*2**n, where y is a float64 and n is an 
large = dtype([('x',float64),('n',int64)])
def aslarge(a):
  assert all(a)
  x,e = frexp(a)
  return rec.fromarrays([x,e],dtype=large)
def large_mul(a,b):
  x,e = frexp(a.x*b.x)
  return rec.fromarrays([x,e+a.n+b.n],dtype=large)
def large_div(a,b):
  x,e = frexp(a.x/b.x)
  return rec.fromarrays([x,e+a.n-b.n],dtype=large)
def large_approx(a):
  return ldexp(a.x,a.n)

class TridiagonalInverse(object):
  '''A representation for the inverse of a tridiagonal matrix.  After O(n) construction, any coefficient can be computed in O(1) time.
  For details, see Gerard Meurant (1992), "A review on the inverse of symmetric tridiagonal and block tridiagonal matrices."'''
  
  def __init__(self,A):
    A = asarray(A)
    n = len(A)
    assert A.shape==(n,2)
    if not n:
      return
    # We follow the notation from Meurant, except that indices begin at zero:
    a = A[:,0]
    b = -A[:,1]
    b = copysign(maximum(abs(b),finfo(b.dtype).tiny),b)
    d = empty_like(a)
    d[n-1] = a[n-1]
    for i in xrange(n-2,-1,-1):
      d[i] = a[i]-b[i]**2/d[i+1]
    e = empty_like(a) # e = \delta from Meurant
    e[0] = a[0]
    for i in xrange(1,n):
      e[i] = a[i]-b[i-1]**2/e[i-1]
    # Precompute partial products needed for theorem, using the large exponent storage defined above
    dd = aslarge(hstack([d,1]))
    ee = aslarge(hstack([e,1]))
    for i in xrange(n-1,-1,-1):
      dd[i] = large_mul(dd[i],dd[i+1])
      ee[i] = large_mul(ee[i],ee[i+1])
    bb = aslarge(hstack([1,b[:-1]]))
    for i in xrange(1,n):
      bb[i] = large_mul(bb[i],bb[i-1])
    dd = dd[1:]
    ee = ee[:-1]
    stuff = large_approx(large_mul(large_mul(bb,bb),large_mul(dd,ee)))
    assert all(stuff[:-1]>stuff[1:])
    # Store
    self.bb = bb
    self.dd = dd
    self.ee = ee

  def diag(self,i):
    return large_approx(large_div(self.dd[i],self.ee[i]))

  def __call__(self,i,j):
    if i > j:
      i,j = j,i
    return large_approx(large_div(large_mul(self.bb[j],self.dd[j]),
                                  large_mul(self.bb[i],self.ee[i])))

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
    if 1:
      B = TridiagonalInverse(A)
    else:
      C = zeros((n,n))
      for i in xrange(n):
        C[i,i] = A[i,0]
      for i in xrange(n-1):
        C[i,i+1] = C[i+1,i] = A[i,1]
      C = linalg.inv(C)
      class Bc(object):
        def __call__(self,i,j):
          return C[i,j]
        def diag(self,i):
          return C[i,i]
      B = Bc()

    def fast_middle(i,xi,k,xk,j):
      '''Compute xj given xi and xk in O(1) time, using the explicit inverse B = A^{-1}'''
      assert -1<=i<j<k<=n
      if 0<=i:
        if k<n:
          # We seek y = a e_i + b e_k s.t. (B y)_i = x(i), (B y)_k = x(k).  That is
          #   B(i,i) a + B(i,k) b = x(i)
          #   B(k,i) a + B(k,k) b = x(k)
          # This 2x2 symmetric matrix is a minor of B, so it can be stably used.
          # Hmm, maybe it isn't so stable after all.  TODO: Try to work with it directly.
          #   B(i,j) = b(i)...b(j-1) d(j+1)...d(n) / e(i)...e(n)   for i <= j
          #   B(i,i) = d(i+1)...d(n) / e(i)...e(n)    
          #   ( B(i,i) B(i,k) ) = ( dd(i+1)/ee(i)  bb(i,k)dd(k+1)/ee(i)  ) = dd(k+1)/ee(i) ( d(i+1)...d(k)    bb(i,k)    )
          #   ( B(i,k) B(k,k) )   (     .            dd(k+1)/ee(k)       )                 (    bb(i,k)    e(i)...e(k-1) )
          #   det = dd(i+1)dd(k)/(ee(i)ee(k)) - bb(i,k)^2 dd(k+1)^2/ee(i)^2
          #       = dd(k+1)^2/ee(i)^2 (d(i+1)...d(k)e(i)...e(k-1) - bb(i,k)^2) 
          #
          #   C = 1/(B(i,i)B(k,k)-B(i,k)^2) (  B(k,k) -B(i,k) )
          #                                 ( -B(i,k)  B(i,i) )
          #   C (1 0)' = (B(k,k)-B(i,k)/(B(i,i)B(k,k)-B(i,k)^2)
          Bik = B(i,k)
          a,b = linalg.solve(((B.diag(i),Bik),(Bik,B.diag(k))),(xi,xk))
          return B(j,i)*a + B(j,k)*b
        else: # k==n
          # We seek y = a e_i s.t. (B y)_i = x(i).  Thus
          return xi*B(j,i)/B.diag(i)
      elif k<n: # -i==i
        return xk*B(j,k)/B.diag(k)
      else: # -i==i, k==n
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
