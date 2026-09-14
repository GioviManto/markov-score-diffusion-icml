"""Scalar (L=1) high-noise exponents with fine quadrature (FFT convolution)."""
import numpy as np
from scipy.special import gamma as G
from scipy.signal import fftconvolve
u = np.linspace(-16, 16, 8001); du = u[1]-u[0]
def gg_pdf(x, beta):
    s = np.sqrt(G(1/beta)/G(3/beta)); return beta/(2*s*G(1/beta))*np.exp(-(np.abs(x)/s)**beta)
def noised(p0, t):
    al = np.exp(-t); de = 1-al**2
    # law of al*a : density (1/al) p0(x/al) on grid u; convolve with N(0,de)
    pa = np.interp(u/al, u, p0, left=0, right=0)/al
    g = np.exp(-u**2/(2*de))/np.sqrt(2*np.pi*de)
    pt = fftconvolve(pa, g, mode='same')*du
    return pt/np.sum(pt*du)
def score(p): return np.gradient(np.log(np.maximum(p,1e-300)), du)
def kurt(p0): m2=np.sum(u**2*p0*du); m4=np.sum(u**4*p0*du); return m4/m2**2-3
ts = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])
print("(1) E_{P_t}(S_t + x)^2 : squared score gap vs unit Gaussian; prediction slope -> -8 (degree-4 leading)")
for name,beta in [("laplace",1.0),("gg beta=1.5",1.5),("gg beta=4",4.0)]:
    p0=gg_pdf(u,beta); k4=kurt(p0); out=[]
    for t in ts:
        pt=noised(p0,t); S=score(pt); m=np.abs(u)<9
        out.append(np.sum(((S+u)**2*pt*du)[m]))
    out=np.array(out); sl=np.diff(np.log(out))/np.diff(ts)
    print(f"  {name:12s} k4={k4:+.3f} local slopes: {np.round(sl,2)}  | gap*exp(8t): {np.round(out*np.exp(8*ts),4)}  | k4^2/24={k4**2/24:.4f}")
print("\n(2) Fisher information, scalar: shape beta at fixed unit variance (pred -8); variance v (pred -4)")
def fisher(pfun, th, t, h=1e-3):
    pp=noised(pfun(th+h),t); pm=noised(pfun(th-h),t); p0=noised(pfun(th),t)
    dl=(np.log(np.maximum(pp,1e-300))-np.log(np.maximum(pm,1e-300)))/(2*h); m=np.abs(u)<9
    return np.sum((dl**2*p0*du)[m])
for name,b0 in [("laplace",1.0),("gg beta=1.5",1.5)]:
    Is=np.array([fisher(lambda b: gg_pdf(u,b), b0, t) for t in ts])
    Iv=np.array([fisher(lambda v: gg_pdf(u/np.sqrt(v),b0)/np.sqrt(v), 1.0, t) for t in ts])
    print(f"  {name:12s} shape slopes {np.round(np.diff(np.log(Is))/np.diff(ts),2)} | I_shape*exp(8t): {np.round(Is*np.exp(8*ts),4)}")
    print(f"  {'':12s} var   slopes {np.round(np.diff(np.log(Iv))/np.diff(ts),2)} | I_var*exp(4t): {np.round(Iv*np.exp(4*ts),4)}")
