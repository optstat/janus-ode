#!/usr/bin/env python3
# vdp_petsc_wheel.py  –  adaptive BE using options (no helpers needed)

from petsc4py import PETSc
import argparse

p = argparse.ArgumentParser()
p.add_argument('--mu', type=float, default=1e3)
p.add_argument('--tf', type=float, default=3e3)
p.add_argument('--dt', type=float, default=1e-3)
p.add_argument('petsc', nargs=argparse.REMAINDER,
               help='PETSc flags after "--", e.g. -- -ts_monitor')
args = p.parse_args()

# Put desired PETSc flags in the option database
opts = PETSc.Options()
opts.insertString('-ts_type beuler')
opts.insertString('-ts_adapt_type basic')
for tok in args.petsc:
    opts.insertString(tok)

mu = args.mu
Y  = PETSc.Vec().createSeq(2)
F  = PETSc.Vec().createSeq(2)
J  = PETSc.Mat().createDense([2,2]);  J.setUp()

def rhs(ts, t, Y, F, mu=mu):
    x1, x2 = Y[0], Y[1]
    F[0] = x2
    F[1] = mu*(1-x1**2)*x2 - x1

def jac(ts, t, Y, A, P, mu=mu):
    x1, x2 = Y[0], Y[1]
    A[0,0]=0; A[0,1]=1
    A[1,0]=-1-2*mu*x1*x2;  A[1,1]=mu*(1-x1**2)
    A.assemble(); P.assemble();  return True

ts = PETSc.TS().create()
ts.setRHSFunction(rhs, F)
ts.setRHSJacobian(jac, J, J)
ts.setTimeStep(args.dt)
ts.setMaxTime(args.tf)
ts.setFromOptions()            # ← parse the options we inserted

Y[:] = (2.0, 0.0)
ts.solve(Y)

print(f"Done: t={ts.time:.3g}  steps={ts.getStepNumber()}  "
      f"h_last={ts.getTimeStep():.3e}  ||Y||₂={Y.norm():.3e}")
