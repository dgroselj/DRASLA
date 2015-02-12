DRASLA
========

DRASLA - Diffusion, Reactions, And Semi-Lagrangian Advection. A C++ code written by [Daniel Groselj](mailto:daniel.grosel@gmail.com).

DRASLA solves reaction-diffusion-advection type equations on a square domain with periodic boundaries. The software is released together with a pseudospectral, two-dimensional Navier-Stokes equation solver that can be used to generate velocity field input data for DRASLA.

A detailed documentation of the project may be found in the `doc/` subdirectory.

Examples
------------

Some examples demonstrating the use of DRASLA are implemented in `examples.h`. Examples demonstrating the use of the Navier-Stokes solver are implemented in `2DTurb/examples2DTurb.h`. 

To run one of the examples, _uncomment_ the call to the corresponding function in `main.cpp` before compiling the program.

Articles using DRASLA
---------

- D. Groselj, F. Jenko, and E. Frey, "How turbulence regulates biodiversity in systems with cyclic competition", [arXiv:1411.4245](http://arxiv.org/abs/1411.4245).
