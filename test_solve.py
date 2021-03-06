#!/usr/bin/env python

from __future__ import division
from other.core import *
from tridiagonal import *
from error import *

def random_system(n,cp=1/3,shift=0):
  # Build a random positive definite tridiagonal matrix via Cholesky factorization
  # A[i,a] = A(i+a,i), same for L
  # A(i,j) = sum_k L(i,k) L(j,k)
  # A[i,0] = A(i,i) = L(i,i)**2 + L(i,i-1)**2 = L[i,0]**2 + L[i-1,1]**2
  # A[i,1] = A(i,i+1) = L(i,i)*L(i+1,i) = L[i,0]*L[i,1]
  L = random.randn(n,2)
  A = empty_like(L)
  if n:
    A[0,0] = L[0,0]**2
    A[-1,1] = 0
  A[1:,0] = L[1:,0]**2+L[:-1,1]**2
  A[:-1,1] = L[:-1,0]*L[:-1,1]
  A[:,0] += shift
  # Construct lo,x,hi such that interesting interactions are likely
  lo,x,hi = sort(random.randn(3,n),axis=0)
  p = random.uniform(size=n)
  for i in xrange(n):
    if p[i]<cp: # Make lo significant
      lo[i],x[i] = x[i],lo[i]
    elif p[i]>1-cp: # Make hi significant
      hi[i],x[i] = x[i],hi[i]
  # Nearly there
  b = tridiagonal_dot(A,x)
  return A,b,lo,hi,x

def expand_tridiagonal(A):
  n = len(A)
  assert A.shape==(n,2)
  M = zeros((n,n))
  if n:
    for i in xrange(n-1):
      M[i,i] = A[i,0] 
      M[i,i+1] = M[i+1,i] = A[i,1] 
    M[-1,-1] = A[-1,0]
  return M

def test_qp():
  tol = 4e-15
  shift = 1e-10
  verbose = 0
  random.seed(71183119)
  for n in range(50)+[100,1000,10000]:
    for i in xrange(10):
      A,b,lo,hi,x = random_system(n,shift=shift)
      if verbose:
        print
        print '-----------------------------------'
        print 'A =\n%s'%A.T
        print 'A =\n%s'%repr(list(map(list,A.T)))
        if 0 and n:
          print 'eigs = %s'%linalg.eigvalsh(expand_tridiagonal(A))
        print 'b = %s'%b
        print 'lo = %s'%lo
        print 'hi = %s'%hi
        print 'x = %s'%x
      x = tridiagonal_qp(A,b,lo,hi)
      if verbose:
        print
        print 'mode = %s'%mode
        print 'x = %s'%x
      e = tridiagonal_qp_error(A,b,lo,hi,x)
      en = maxabs(e)
      if verbose or en>=tol:
        print 'n %d, lo %d, hi %d'%(n,sum(abs(x-lo)<1e-10),sum(abs(x-hi)<1e-10))
        print 'e = %g\n%s'%(en,asarray(e))
      assert en<tol

if __name__=='__main__':
  set_printoptions(linewidth=230,precision=4)
  test_qp()
