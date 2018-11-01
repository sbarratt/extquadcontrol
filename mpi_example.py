from mpi4py import MPI
from extquadcontrol import ExtendedQuadratic, dp_finite_mpi
import numpy as np


# time mpirun -n 6 python mpi_example.py

comm = MPI.COMM_WORLD

n, m = 25,50
K = 3
N = 100
T = 25

As = .1*np.random.randn(1,1,n,n)
Bs = np.random.randn(1,1,n,m)
cs = np.zeros((1,1,n))
gs = [ExtendedQuadratic(np.eye(n+m),np.zeros(n+m),0) for _ in range(K)]
g_T = [ExtendedQuadratic(np.eye(n),np.zeros(n),0) for _ in range(K)]
Pi = np.eye(K)
def sample(t, N):
    A = np.zeros((N,K,n,n)); A[:] = As
    B = np.zeros((N,K,n,m)); B[:] = Bs
    c = np.zeros((N,K,n)); c[:] = cs
    g = [gs for _ in range(N)]
    return A,B,c,g,Pi

import time
start = time.time()
result = dp_finite_mpi(sample, g_T, T, N, comm)
end = time.time()
if comm.Get_rank() == 0:
	print ("time: %.3f" % (1000*(end-start)), "ms")
	Vs, Qs, policies = result