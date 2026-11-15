from aemo_release import np,R,feasible,exact
def ray(b,z,bq,bal):
    # Invert the scalar envelope; exact tests below certify the stored vector.
    a = np.array([float(v) for v in bal])
    r = float(R)
    width = np.sqrt(a*(r-1)/r)
    lo = np.maximum(0,b-width)
    hi = np.where(a<=r*(r-1)*b*b, b+width, np.sqrt(a+r*b*b))
    direction = z-b
    ratios = np.ones_like(b)
    up,down = direction>0,direction<0
    ratios[up] = (hi[up]-b[up])/direction[up]
    ratios[down] = (lo[down]-b[down])/direction[down]
    t = min(1.,max(0.,float(ratios.min())))
    for contraction in [1.,1.-1e-12,1.-1e-10,1.-1e-8,1.-1e-6,1.-1e-4]:
        w = t*contraction
        p = z if w==1 else (1-w)*b+w*z
        if feasible(bq,exact(p),bal): return p,w
    # Rounding very close to the anchor can mask the effect of the backoff.
    l,h=0.,t
    for _ in range(45):
        w=(l+h)*0.5
        if feasible(bq,exact((1-w)*b+w*z),bal): l=w
        else: h=w
    return (1-l)*b+l*z,l
