"""Exact arithmetic check for the common-state region/system example."""
from fractions import Fraction as F

def gamma(b,p):
    r=F(40401,40000)
    return p*p-r*b*b+max(0,r*b-p)**2/(r-1)

def check():
    b=(20,10,10); y0=(120,10,110); z=(23,12,11)
    allowances=[F(401,40000)*(q-y)**2 for q,y in zip(b,y0)]
    assert allowances==[F(401,4),0,F(401,4)]
    for h in (1,2):
        p=(20+h,10,10+h)
        assert all(gamma(i,j)<=a for i,j,a in zip(b,p,allowances))
        assert sum((i-j)**2 for i,j in zip(z,p)) == (8 if h==1 else 6)
    anchor=(10,10,0); alternative=(12,0,11); release=(10,10,2)
    assert all(min(a,b)<=c<=max(a,b) for a,b,c in zip(anchor,alternative,release))
    assert (sum(release),release[0],sum(release[1:])) == (22,10,12)
    print('PASS exact common-state example and nonunique-witness counterexample')

if __name__=='__main__':
    check()

