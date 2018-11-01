# extquadcontrol

Code accompanying the paper 'Stochastic Control with Affine Dynamics and Extended Quadratic Costs'.

### Installing conda environment

We use [conda](https://conda.io/miniconda.html) for python package management.
To install our environment, run the following commands while in the repository:
```
$ conda env create -f environment.yml
$ conda activate gen_lqr
```

### Running tests
To run tests, run (with the gen_lqr environment activated):
```
$ python extquadcontrol.py
```
These should pass.

### Running the Examples
Examples from 6.1-6.9 in the paper are in the iPython notebooks.
To view them, run (with the gen_lqr environment activated):
```
$ jupyter notebook .
```

### MPI Implementation
To test the MPI implementation, run (with the gen_lqr environment activated):
```
mpirun -n 6 python mpi_example.py
```
This runs the MPI implementation described in the paper with 6 threads. (Try 12, it just works!)# extquadcontrol
