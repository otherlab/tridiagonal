A linear time direct solver for convex tridiagonal quadratic programs with bound constraints
============================================================================================

The title says it all.  The algorithm is similar to convex hull of a simple polygon,
and even more similar in spirit to the linear time algorithm for shortest paths in
polygons in the plane.  For details, see `tridiagonal.tex`.  Code and paper are both
BSD licensed (see `LICENSE`).

### Dependencies

The core algorithm is written in C++, with test routines in Python.  To interface between
C++ and Python, we depend on

* [other/core](https://github.com/otherlab/core): Otherlab core utilities and Python bindings

### Setup

1. Install `other/core` and the other dependencies.
2. Setup and build the code:

        cd tridiagonal
        $OTHER/core/build/setup
        scons -j 5

3. Test:

        py.test
