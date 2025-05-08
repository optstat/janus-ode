#!/usr/bin/env python3
# mpi_petsc_test.py  –  MPI + PETSc sanity check

from mpi4py import MPI
from petsc4py import PETSc
import petsc4py                       # <-- for __version__
import numpy as np
import torch
comm  = MPI.COMM_WORLD
rank  = comm.rank
size  = comm.size

# ----- tiny 2×2 solve on each rank ------------------------------------
A = PETSc.Mat().createAIJ([2, 2], comm=PETSc.COMM_SELF)
A.setValuesCSR([0, 2, 4],            # indptr
               [0, 1, 0, 1],         # indices
               [3.0, 1.0, 1.0, 2.0]) # values
A.assemble()

b = PETSc.Vec().createSeq(2)
b.set(1.0 + rank)                    # different RHS per rank

ksp = PETSc.KSP().create(PETSc.COMM_SELF)
ksp.setOperators(A)
ksp.setType('cg')
ksp.setTolerances(rtol=1e-10)
ksp.solve(b, b)                      # reuse b as x

x_local = b.getArray()
xs = comm.gather(x_local, root=0)

# ----- rank 0 prints summary ------------------------------------------
if rank == 0:
    print(f"petsc4py {petsc4py.__version__}, "
          f"PETSc {PETSc.Sys.getVersion()}, "
          f"MPI ranks {size}")
    for r, vec in enumerate(xs):
        print(f"  rank {r}: x = {vec}")
