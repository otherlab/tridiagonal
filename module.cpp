// A linear time direct solver for tridiagonal quadratic programs with bound constraints

#include <other/core/array/Array.h>
#include <other/core/python/module.h>
#include <other/core/python/wrap.h>
#include <other/core/vector/Vector.h>
#include <other/core/structure/Tuple.h>
namespace {

using namespace other;
typedef double T;
using std::ostream;
using std::cout;
using std::endl;

// After shifting to eliminate b, our energy has the form
//
//   E = 1/2 x'Ax
//
// Decompose A as A = LDL', where L is unit lower bidiagonal and D is positive diagonal.  We have
//   A(i,j) = L(i,k) D(k) L(j,k)
//   A(i,i) = D(i) + D(i-1) L(i,i-1)^2
//   A(i,i+1) = D(i) L(i+1,i)
//   E = 1/2 (L'x)' D (L'x) = 1/2 D(i) (x(i) + L(i+1,i) x(i+1))^2
//
// To show monotonicity more clearly we store the offdiagonal of L negated.  Thus, energy Eij has the form
//
//   Eij = 1/2 di (xi - lji xj)^2
//
// Now say we have Eij, Ejk and wish to eliminate j.  Mathematica gives
//
//   Eik = 1/2 di (xi - lij xj)^2 + 1/2 dj (xj - ljk xk)^2
//   xj = (di lij xi + dj ljk xk) / (dj + di lij^2)
//   Eik = 1/2 di dj / (dj + di lij^2) (xi - lij ljk x2)^2

// Energy term connecting i to j.
struct Energy {
  OTHER_DEBUG_ONLY(int i,j;) // In debug mode, keep track of which indices this energy connects for assertion purposes.
  T d; // diagonal di = Di
  T l; // negated lower entry lij = -Lji

  OTHER_DEBUG_ONLY(Energy() : i(-2), j(-2) {})
};

static inline ostream& operator<<(ostream& output, const Energy E) {
  return output << '(' OTHER_DEBUG_ONLY(<<E.i<<','<<E.j<<',') << E.d << ',' << E.l << ')';
}

inline T solve(const Energy ij, const Energy jk, const T xi, const T xk) {
  assert(ij.j==jk.i);
  return (ij.d*ij.l*xi+jk.d*jk.l*xk)/(jk.d+ij.d*sqr(ij.l));
}

inline Energy reduce(const Energy ij, const Energy jk) {
  assert(ij.j==jk.i);
  Energy ik;
  OTHER_DEBUG_ONLY(ik.i = ij.i; ik.j = jk.j;)
  ik.d = ij.d*jk.d/(jk.d+ij.d*sqr(ij.l));
  ik.l = ij.l*jk.l;
  return ik;
}

// Compute the LDL factorization of a SPD tridiagonal A
Array<Energy> cholesky_factor(RawArray<const Vector<T,2>> A) {
  const int n = A.size();
  Array<Energy> E(n,false);
  if (n) {
    OTHER_DEBUG_ONLY(E[0].i = 0;)
    E[0].d = A[0].x;
    for (int i=0;i<n-1;i++) {
      OTHER_DEBUG_ONLY(E[i].j = E[i+1].i = i+1;)
      E[i].l = -A[i].y/E[i].d;
      E[i+1].d = A[i+1].x-E[i].d*sqr(E[i].l);
    }
    OTHER_DEBUG_ONLY(E[n-1].j = n;)
    E[n-1].l = 0;
  }
  return E;
}

// Solve A x = b given the LDL factorization of A
Array<T> cholesky_solve(RawArray<const Energy> E, RawArray<const T> b) {
  const int n = b.size();
  Array<T> x(n,false);
  if (n) {
    // x = D^{-1} L^{-1} b
    T xp = b[0];
    for (int i=1;i<n;i++) {
      const T xn = b[i]+E[i-1].l*xp;
      x[i-1] = xp/E[i-1].d;
      xp = xn;
    }
    x[n-1] = xp/E[n-1].d;
    // x = L^{-T} x
    for (int i=n-2;i>=0;i--)
      x[i] += E[i].l*x[i+1];
  }
  return x;
}

// Maintain the product of a queue of semigroup (Energy) elements
// The queue is theoretically represented as two stacks, but we store them together as one array
// split in the middle at a pivot index.
struct SemigroupQueue {
  struct Entry {
    int i;
    Energy self; // E(i,j)
    Energy prod; // E(i,pivot) for i < pivot, meaningless if i >= pivot
  };
  Energy left; // E(prefix,data[start].i): prefix to the start of the queue
  const Array<Entry> data;
  Energy right; // E(pivot,data[stop-1].j): pivot to the right of the end of the queue
  int start, stop; // First valid index, last valid index+1, indexing into data.
  int pivot; // Pivot index as a system index.  It does not index into data since the pivot element might not exist in data.

  SemigroupQueue(const int n, const Energy info)
    : data(n), start(0), stop(1), pivot(0) {
    OTHER_DEBUG_ONLY(left.i = -1; left.j = 0;)
    right = info;
    assert(info.i==left.j);
    assert(info.j==right.j);
    left.d = 1;
    left.l = 0;
    data[0].i = 0;
    data[0].self = info;
  }

  int size() const {
    return stop-start;
  }

  Tuple<int,Energy> front() const {
    assert(size());
    const auto& s = data[start];
    assert(s.i<=pivot);
    return tuple(s.i, s.i==pivot ? right : reduce(s.prod,right));
  }

  Tuple<int,Energy> from_back(int s) const {
    assert(unsigned(s)<unsigned(size()));
    const auto& e = data[stop-1-s];
    return tuple(e.i,e.self);
  }

  void append(const int i, const Energy info) {
    assert(i >= pivot);
    assert(info.i==i && info.j==i+1);
    if (size())
      right = reduce(right,info);
    else {
      right = info;
      pivot = i;
    }
    Entry e;
    e.i = i;
    e.self = info;
    data[stop++] = e;
  }

  void pop() {
    assert(size());
    stop--;
    Energy& next = size() ? data[stop-1].self : left;
    next = reduce(next,data[stop].self);
  }

  void pop_left() {
    assert(size());
    left = data[start++].self;
    if (size() && data[start].i > pivot) {
      // We've moved past the pivot, so spend O(n) to move it to the end.  This amortizes to O(1) by charging to append.
      pivot = data[stop-1].i;
      right = data[stop-1].self;
      if (size()>1) {
        assert(data[stop-2].self.j==pivot);
        data[stop-2].prod = data[stop-2].self;
        for (int i=stop-3;i>=start;i--)
          data[i].prod = reduce(data[i].self,data[i+1].prod);
      }
    }
  }
};

// One half of the main algorithm loop.  If sign<0, (lo,hi),(lower,upper) are really (hi,lo),(upper,lower).
template<int sign> static inline void extend(RawArray<const Energy> E, RawArray<const T> lo, RawArray<const T> hi, Array<Tuple<int,T>>& prefix, SemigroupQueue& lower, SemigroupQueue& upper, const int k) {
  BOOST_STATIC_ASSERT(sign==1 || sign==-1);
  const int n = E.size();
  // Reduce lower as far as necessary
  const T xk = k<n ? lo[k] : 0;
  while (lower.size()) {
    const auto Ejk = lower.from_back(0);
    const int j = Ejk.x;
    T xi;
    Energy Eij;
    if (lower.size()>=2) {
      auto e = lower.from_back(1);
      Eij = e.y;
      xi = lo[e.x];
    } else {
      assert(lower.left.i == prefix.back().x);
      Eij = lower.left;
      xi = prefix.back().y;
    }
    const T xj = solve(Eij,Ejk.y,xi,xk);
    if (sign>0 ? xj<=lo[j] : xj>=lo[j]) // If the constraint is necessary, stop eroding
      break;
    else // Otherwise, disclose the lo[j] constraint and keep reducing
      lower.pop();
  }
  // If we've exhausted lower, erode upper from the left
  if (!lower.size())
    while (upper.size()) {
      const T xi = prefix.back().y;
      const auto Ejk = upper.front();
      const int j = Ejk.x;
      assert(Ejk.y.j==k);
      const T xj = solve(upper.left,Ejk.y,xi,xk);
      if (sign>0 ? xj<=hi[j] : xj>=hi[j]) // If the hi[j] constraint doesn't conflict, stop eroding
        break;
      else { // If the hi[j] constraint does conflict, the shared path must include x[j] = hi[j].
        prefix.append_assuming_enough_space(tuple(j,hi[j]));
        upper.pop_left();
        lower.left = upper.size() ? reduce(upper.left,upper.front().y) : upper.left;
      }
    }
}

// Find x minimizing E = 1/2 x'Ax - b'x subject to lo <= x <= hi, where A is tridiagonal and symmetric positive definite.
// The algorithm is a linear time active set method analogous to convex hull of sorted points.
//
// Parameters
// ----------
// A : The SPD tridiagonal matrix A given as an (n,2) array.  A[0] is the diagonal and A[1,:-1] is the offdiagonal
// b : Linear term of the objective
// lo,hi : Bounds on x, which may be infinite
//
// Returns
// -------
// x : Solution
static Array<T> tridiagonal_qp(RawArray<const Vector<T,2>> A, RawArray<const T> b, RawArray<const T> lo_, RawArray<const T> hi_) {
  // Check consistency
  const int n = A.size();
  OTHER_ASSERT(b.size()==n);
  OTHER_ASSERT(lo_.size()==n);
  OTHER_ASSERT(hi_.size()==n);
  if (!n)
    return Array<T>();
  for (const int i : range(n)) {
    if (lo_[i] > hi_[i])
      throw ValueError("tridiagonal_qp: lo > hi");
    if (A[i].x < 0)
      throw ValueError("The diagonal of A must be strictly positive");
  }

  // Offset x so that b = 0
  const auto E = cholesky_factor(A);
  for (const int i : range(n))
    if (E[i].d <= 0)
      throw ArithmeticError(format("nonpositive diagonal %d = %g in Cholesky factorization",i,E[i].d));
  const auto base = cholesky_solve(E,b),
             lo = (lo_-base).copy(),
             hi = (hi_-base).copy();

  // Adjust signs so that E[:].l >= 0
  const Array<bool> flip(n+1,false);
  flip[0] = false;
  for (int i=0;i<n;i++) {
    if (flip[i])
      vec(-hi[i],-lo[i]).get(lo[i],hi[i]);
    flip[i+1] = flip[i]^(E[i].l<0);
    E[i].l = abs(E[i].l);
  }

  // Our internal state consists of
  //
  // 1. A list of known x values common to both lower and upper paths.
  // 2. An optimal lower path ending at x[k] = lo[k], represented as a deque of indices i where x[i] = lo[i].
  // 3. An optimal upper path analogous to lower but for x[k] = hi[k] and x[i] = hi[i].
  Array<Tuple<int,T>> prefix;
  prefix.preallocate(n+1);
  prefix.append_assuming_enough_space(tuple(-1,T(0)));
  SemigroupQueue lower(n,E[0]),
                 upper(n,E[0]);

  // On to the core algorithm
  for (const int k : range(1,n+1)) {
    extend<+1>(E,lo,hi,prefix,lower,upper,k); // Erode lower
    extend<-1>(E,hi,lo,prefix,upper,lower,k); // Erode upper
    // Advance
    if (k<n) {
      lower.append(k,E[k]);
      upper.append(k,E[k]);
    }
  }

  // If nothing was constrained, we're done
  if (prefix.size()==1)
    return base;

  // Expand prefix into a full solution
  Array<T> x(n,false);
  T xi = 0;
  for (int a=0;a<prefix.size();a++) {
    const int i = prefix[a].x,
              j = a+1<prefix.size() ? prefix[a+1].x : n;
    const T xj = a+1<prefix.size() ? base[j]+(flip[j]?-prefix[a+1].y:prefix[a+1].y) : 0;
    if (j-i > 1) {
      base.slice(i+1,j) = b.slice(i+1,j);
      if (i >= 0)
        base[i+1] -= A[i].y*xi;
      base[j-1] -= A[j-1].y*xj;
      x.slice(i+1,j) = cholesky_solve(cholesky_factor(A.slice(i+1,j)),base.slice(i+1,j));
    }
    if (j < n)
      x[j] = xj;
    xi = xj;
  }
  return x;
}

}

OTHER_PYTHON_MODULE(tridiagonal_core) {
  OTHER_FUNCTION(tridiagonal_qp)
}
