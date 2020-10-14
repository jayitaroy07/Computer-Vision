import cv2
import numpy as np
import sys

def lscl(image):
    a = np.min(image)
    b = np.max(image)
    target = ((image-a)/(b-a))*255
    return target

# def hist(image):
#     n=image.shape[0]*image.shape[1]
#     h = cv2.calcHist([image], [0], None, [256], [0,256])
#     f = np.zeros(256, dtype='int32')
#     f[0] = h[0,0]
#     for i in range(1,256):
#         f[i] = f[i-1] + h[i,0]
#     g = np.zeros(256, dtype='int32')
#     g[0] = f[0]*256/(2*n)
#     for i in range(1,256):
#         g[i] = (f[i-1]+f[i])*256./(2*n)
#     target = image.copy()
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             target[i,j] = g[image[i,j]]
#     return target


# read arguments
if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg output.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]

# check the correctness of the input parameters
if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

# read image
inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()
# cv2.imshow("input image: " + name_input, inputImage)

# check for color image and change w1, w2, h1, h2 to pixel locations 
rows, cols, bands = inputImage.shape
if(bands != 3) :
    print("Input image is not a standard color image:", inputImage)
    sys.exit()

W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))

# The transformation should be applied only to
# the pixels in the W1,W2,H1,H2 range.
# The following code goes over these pixels

# tmp1 = np.copy(inputImage)
# for i in range(H1, H2+1) :
#     for j in range(W1, W2+1) :
#         b, g, r = inputImage[i, j]
#         gray = round(0.3*r + 0.6*g + 0.1*b)
#         tmp1[i, j] = [gray, gray, gray]
# cv2.imshow("replace_gray", tmp1)

# Slicing can be used for similar things
# In this example the red channel is zeroed out
tmp2 = np.copy(inputImage)
target = tmp2[H1: H2+1, W1: W2+1]
target_lab=cv2.cvtColor(target,cv2.COLOR_BGR2Lab)
lmatrix=target_lab[:,:,0]
target_lscl=lscl(lmatrix)
# target_hist=cv2.equalizeHist(lmatrix)
# target_hist=hist(lmatrix)
target_lab[:,:,0]=target_lscl
target_bgr=cv2.cvtColor(target_lab,cv2.COLOR_Lab2BGR)
tmp2[H1: H2+1, W1: W2+1]=target_bgr




# window_height = H2 - H1 + 1
# window_width = W2 -W1 + 1
# window = np.zeros([window_height, window_width],dtype=np.uint8)
# tmp2[H1: H2+1, W1: W2+1, 2] = window
# cv2.imshow("target", tmp2)

# saving the output - save the gray window image
cv2.imwrite(name_output, tmp2)

# wait for key to exit
# cv2.waitKey(0)
# cv2.destroyAllWindows()


