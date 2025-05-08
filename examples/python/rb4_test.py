import math, torch, inspect, sys
Tensor = torch.Tensor

# ── Hard-wired, minimal Rosenbrock-W(4) stepper ───────────────────────
A = torch.tensor([[0.    , 0.   , 0.   , 0.   ],
                  [0.386 , 0.   , 0.   , 0.   ],
                  [0.146 , 0.414, 0.   , 0.   ],
                  [0.063 ,-0.24 , 0.62 , 0.   ]])
C = torch.tensor([[0.25  ,-0.25 , 0.   , 0.   ],
                  [0.1043,-0.1043, 0.   , 0.   ],
                  [0.1035,-0.1035, 0.   , 0.   ],
                  [0.361 ,-0.556 , 0.194, 0.   ]])
M = torch.tensor([0.25 ,0,0.5 ,0.25 ])
M2= torch.tensor([0.20 ,0,0.5 ,0.30 ])
ALPHA=torch.tensor([0.0 ,0.386,0.56 ,0.95])
GAMMA=0.25;  ORDER=4; ERR_EXP=1/5

def solve(f, jac, t0, y0, tf, rtol=1e-6, atol=1e-9, h0=1e-3):
    t, y, h, J = t0, y0.clone(), h0, jac(t0, y0)
    n=0
    while t < tf:
        if t+h > tf: h = tf-t
        I = torch.eye(len(y)); LU = torch.linalg.lu_factor(I - GAMMA*h*J)
        k=[]
        for i in range(4):
            y_inc = sum(A[i,j]*k[j] for j in range(i)) if i else torch.zeros_like(y)
            rhs   = h*f(t+ALPHA[i]*h, y+y_inc)
            coupl = sum(C[i,j]*k[j] for j in range(i)) if i else torch.zeros_like(y)
            rhs  += h*GAMMA*J@coupl
            k.append(torch.linalg.lu_solve(*LU, rhs.unsqueeze(1)).squeeze(1))
        y_new = y + sum(M[i]*k[i] for i in range(4))
        err   =     sum((M[i]-M2[i])*k[i] for i in range(4))
        err_n = torch.norm(err/(atol+rtol*y_new.abs()))/math.sqrt(len(y))
        if err_n>1: h*=max(0.4,0.9*err_n**(-ERR_EXP)); continue
        t += h; y=y_new; J=jac(t,y); n+=1
        h = min(h*min(10,0.9*err_n**(-ERR_EXP)), tf-t)
    print("steps",n,"final h",h)
    return y
# ── Van-der-Pol test ─────────────────────────────────────────────────
mu=5.
def vdp(t,y): return torch.stack([y[1], mu*(1-y[0]**2)*y[1]-y[0]])
def jac_v(t,y): return torch.tensor([[0.,1.],
                                     [-2*mu*y[0]*y[1]-1., mu*(1-y[0]**2)]])
print("initial h 1e-3")
solve(vdp,jac_v,0.,torch.tensor([2.,0.]),20.)
