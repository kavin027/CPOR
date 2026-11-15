"""Two shrinkage coefficients constrained by all observed nodes."""
import numpy as np

def partition(S):
    target_size = (S.shape[1] + 1) // 2
    selected = set()
    supports = [set(np.flatnonzero(row > 0)) for row in S]
    for s in sorted(supports, key=len, reverse=True):
        if selected.isdisjoint(s) and len(selected) + len(s) <= target_size:
            selected.update(s)
    for j in range(S.shape[1]):
        if len(selected) == target_size:
            break
        selected.add(j)
    return np.array(sorted(selected), dtype=int)

def clip(vertices, a, b, cap):
    """Intersect a convex polygon (possibly a segment) with one half-plane."""
    out = []
    start = vertices[-1]
    ds = start[0]*a + start[1]*b - cap
    for end in vertices:
        de = end[0]*a + end[1]*b - cap
        if (de > 0) != (ds > 0):
            f = ds / (ds-de)
            out.append((start[0]+f*(end[0]-start[0]), start[1]+f*(end[1]-start[1])))
        if de <= 0:
            out.append(end)
        start, ds = end, de
    return out

def optimum(D, lo, hi, inv_var):
    vertices = [(0.,0.),(1.,0.),(1.,1.),(0.,1.)]
    # Only inequalities violated anywhere on the initial square can bind.
    for normals, caps in ((D, hi),(-D,-lo)):
        for i in np.flatnonzero(np.maximum(normals,0.).sum(axis=1) > caps):
            a,b = normals[i]
            vertices = clip(vertices, a, b, caps[i])
            if not vertices:
                raise ArithmeticError('Lost feasible origin in polygon clipping')
    starts = np.array(vertices)
    edges = np.roll(starts,-1,axis=0) - starts
    H = D.T @ (inv_var[:,None]*D)
    den = (edges @ H * edges).sum(axis=1)
    num = ((1.-starts) @ H * edges).sum(axis=1)
    f = np.clip(np.divide(num, den, out=np.zeros_like(num), where=den>0),0.,1.)
    candidates = starts + f[:,None]*edges
    objectives = (((candidates-1.) @ D.T)**2) @ inv_var
    return np.clip(candidates[objectives.argmin()],0.,1.)

def grouped(SA, mask, b, ub, z, u, bal, bq, inv_var, ns, R):
    if not any(bal):
        return b.copy(), ub.copy(), [0.,0.], 'zero'
    if ns['feasible'](bq,ns['exact'](z),bal):
        return z.copy(), u.copy(), [1.,1.], 'target'
    credit = np.array([float(a) for a in bal])
    width = np.sqrt(credit*(R-1)/R)
    lo = np.maximum(0., b-width)
    hi = np.where(credit<=R*(R-1)*b*b,b+width,np.sqrt(credit+R*b*b))
    d0 = SA @ (u[mask]-ub[mask])
    # Close the second direction on the stored endpoints (floating reconstruction).
    D = np.column_stack((d0,(z-b)-d0))
    weights = optimum(D,lo-b,hi-b,inv_var)
    def point(w):
        return (1.-w[1])*b + w[1]*z + (w[0]-w[1])*d0
    def valid(p):
        return np.all(p>=0) and ns['feasible'](bq,ns['exact'](p),bal)
    repair = 'none'
    for f in (1.,1.-1e-12,1.-1e-10,1.-1e-8,1.-1e-6,1.-1e-4):
        w = weights*f
        p = point(w)
        if valid(p):
            repair = 'none' if f==1. else 'backoff'
            break
    else:
        low, high = 0., 1.
        for _ in range(45):
            f = (low+high)*.5
            if valid(point(weights*f)):
                low = f
            else:
                high = f
        w = weights*low
        p = point(w)
        repair = 'bisect' if low>0 else 'anchor'
    witness = (1.-w[1])*ub + w[1]*u
    witness[mask] = (1.-w[0])*ub[mask]+w[0]*u[mask]
    return p,witness,w,repair
