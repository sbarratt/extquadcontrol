# extquadcontrol

Code accompanying the paper 'Stochastic Control with Affine Dynamics and Extended Quadratic Costs'.

### Installation

To install the `extquadcontrol` package, from the main directory, run:
```
$ pip install .
```
For running tests and examples, please 

### Running tests
To run tests, run:
```
$ python -m unittest
```
These should pass. If they do not, please file an issue

### Running the Examples
Examples from 6.1-6.9 in the paper are in the iPython notebooks in the `examples` folder.
To view them, run:
```
$ jupyter notebook .
```

### MPI Implementation
Please install MPI and mpi4py:
```
$ sudo apt install mpich
$ pip install mpi4py
```

To test the MPI implementation, navigate to the `examples` folder and run:
```
mpirun -n 6 python mpi_example.py
```
This runs the MPI implementation described in the paper with 6 threads. (Try 12, it just works!)
