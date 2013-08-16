#!/usr/bin/env python

from __future__ import division
from solve import *

def random_system(n,cp=1/3):
  # Build a random positive definite tridiagonal matrix via Cholesky factorization
  # A[i,a] = A[i+a,i], same for L
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

def maxabs(x):
  x = asarray(x)
  return absolute(x).max() if x.size else 0

def test_inverse():
  random.seed(81731)
  for n in 0,1,2,10,20:
    for s in xrange(2):
      A,_,_,_,_ = random_system(n)
      B = TridiagonalInverse(A)
      for i in xrange(n):
        e = zeros(n)
        e[i] = 1
        Bi = linalg.solveh_banded(A.T,e,lower=True)
        assert allclose(Bi[i],B.diag(i))
        for j in xrange(n):
          assert allclose(Bi[j],B(i,j))

def test_qp():
  tols = {'fast':2e-6,
          'slow':3e-9,
          'debug':1e-10}
  verbose = 0
  random.seed(71183119)
  for n in xrange(20):
    for i in xrange(10):
      A,b,lo,hi,x = random_system(n)
      if verbose:
        print
        print '-----------------------------------'
        print 'A =\n%s'%A.T
        print 'A =\n%s'%repr(list(map(list,A.T)))
        print 'b = %s'%b
        print 'lo = %s'%lo
        print 'hi = %s'%hi
        print 'x = %s'%x
      for mode in 'slow fast'.split():
        x = tridiagonal_qp(A,b,lo,hi,mode=mode)
        if verbose:
          print
          print 'x = %s'%x
        e = tridiagonal_qp_error(A,b,lo,hi,x)
        en = maxabs(e)
        tol = tols[mode]
        if verbose or en>=tol:
          print 'mode = %s'%mode
          print 'n %d, lo %d, hi %d'%(n,sum(abs(x-lo)<1e-10),sum(abs(x-hi)<1e-10))
          print 'e = %g\n%s'%(en,asarray(e))
        assert en<tol

if __name__=='__main__':
  set_printoptions(linewidth=240,precision=4)
  test_qp()
