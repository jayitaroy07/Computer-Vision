import sys
import cv2
import numpy as np

if len(sys.argv)<3:
    print("Wrong usage")

img=cv2.imread(sys.argv[1])
c=float(sys.argv[2])

b=0
h=img.shape[0]
w=img.shape[1]

d=0.21
a=(c-d)*2/w
v0=0
u0=0
f=c

def getuv(p):
    [x,y]=p
    print(x,y,a,b,f)
    u=u0+f*x/(a*x+b*y+d)
    v=v0+f*y/(a*x+b*y+d)
    return [u,v]

rec=np.array([
    [10,20],
    [30,40],
    [70,55],
    [5,2]
], dtype=np.float32)

uv = [getuv(p) for p in rec]
uv=np.array(uv, dtype=np.float32)
M = cv2.getPerspectiveTransform(rec, uv)
warp = cv2.warpPerspective(img, M, (w, h))
cv2.imshow('out',warp)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('21.png', warp)

a=0
b=(c-d)*2/h
def getuv2(p):
    [x,y]=p
    print(x,y,a,b,f)
    u=u0+f*x/(a*x+b*y+c)
    v=v0+f*y/(a*x+b*y+c)
    return [u,v]

rec=np.array([
    [10,20],
    [30,40],
    [70,55],
    [5,2]
], dtype=np.float32)

uv = [getuv(p) for p in rec]
uv=np.array(uv, dtype=np.float32)
M = cv2.getPerspectiveTransform(rec, uv)
warp = cv2.warpPerspective(img, M, (w, h))
cv2.imshow('out',warp)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('22.png', warp)
