import sys
import numpy as np
import cv2

if len(sys.argv)<8:
    print("Wrong usage")

img=cv2.imread(sys.argv[1])
f=float(sys.argv[2])
u0=float(sys.argv[3])
v0=float(sys.argv[4])
a=float(sys.argv[5])
b=float(sys.argv[6])
c=float(sys.argv[7])

def getuv(p):
    [x,y]=p
    print(x,y,a,b,f)
    u=u0+f*x/(a*x+b*y+c)
    v=v0+f*y/(a*x+b*y+c)
    return [u,v]

h=img.shape[0]
w=img.shape[1]

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

cv2.imwrite('1.png', warp)
