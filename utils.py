## average edge strength 
import glob, os
import math
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt 
import cv2
from scipy.ndimage import gaussian_filter
from skimage import feature
import scipy.ndimage as ndi
import scipy


def patch_to_center(input_img, sqsize=896):
    empty_ = np.zeros((sqsize, sqsize))
    imgs_ = input_img.shape

    empty_[int(sqsize/2)-int(imgs_[0]/2): int(sqsize/2)+int(imgs_[0]/2), 
           int(sqsize/2)-int(imgs_[1]/2): int(sqsize/2)+int(imgs_[1]/2)] = input_img

    return empty_




def gaussian_kernel(size, sigma=1, verbose=False):
 
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
 
    kernel_2D *= 1.0 / kernel_2D.max()
 
    if verbose:
        plt.imshow(kernel_2D, interpolation='none',cmap='gray')
        plt.title("Image")
        plt.show()
 
    return kernel_2D

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def convolution(image, kernel, average=False, verbose=False):
 
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))
 
    print("Kernel Shape : {}".format(kernel.shape))
 
    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()

def gaussian_blur(image, kernel_size, verbose=False):
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size), verbose=verbose)
    return convolution(image, kernel, average=True, verbose=verbose)

def gauss2Dfilter(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask -
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def CustomCanny(img):
    """ 
    The steps of the algorithm are as follows:

    1 Smooth the image using a Gaussian with sigma width.

    2 Apply the horizontal and vertical Sobel operators to get the gradients
      within the image. The edge strength is the norm of the gradient.

    3 Thin potential edges to 1-pixel wide curves. First, find the normal
      to the edge at each point. This is done by looking at the
      signs and the relative magnitude of the X-Sobel and Y-Sobel
      to sort the points into 4 categories: horizontal, vertical,
      diagonal and antidiagonal. Then look in the normal and reverse
      directions to see if the values in either of those directions are
      greater than the point in question. Use interpolation to get a mix of
      points instead of picking the one that's the closest to the normal.

    4 Perform a hysteresis thresholding: first label all points above the
      high threshold as edges. Then recursively label any point above the
      low threshold that is 8-connected to a labeled point as an edge.
      """
    return mask

"""
# Compute the Canny filter for two values of sigma
edges1 = feature.canny(im)
edges2 = feature.canny(im, sigma=3)
Process of Canny edge detection algorithm

The Process of Canny edge detection algorithm can be broken down to 5 different steps:

1 Apply Gaussian filter to smooth the image in order to remove the noise
2 Find the intensity gradients of the image
3 Apply non-maximum suppression to get rid of spurious response to edge detection
4 Apply double threshold to determine potential edges
5 Track edge by hysteresis: Finalize the detection of edges by suppressing all
the other edges that are weak and not connected to strong edges.
"""

a = nib.load('./OFF/PD/au70_PD_OFF.nii.gz').get_fdata()
a = a/a.max()
max_ = np.asarray(a.shape).max()
s = a[:,:,0]
#s = patch_to_center(s, sqsize=max_)
#print(s[120,120], s.shape, s.dtype)

E = feature.canny(s, sigma=0.6)
E = E.astype(np.float64)
print(E)

#Gx -----> [-1 -1 -1; 0 0 0; 1 1 1]
#Gy -----> [-1 0 1; -1 0 1; -1 0 1]


Gx = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
Gy = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
#sobel_x = np.array([[ -1, 0, 1],
#                    [ -2, 0, 2],
#                    [ -1, 2, 1]])
#sobel_y = np.array([[ -1, -2, -1],
#                    [ 0, 0, 0],
#                    [ 1, 2, 1]])
#filteredx = cv2.filter2D(s, -1, sobel_x)
#filteredy = cv2.filter2D(s, -1, sobel_y)
filtx = cv2.filter2D(s, -1, Gx)
filty = cv2.filter2D(s, -1, Gy)

plt.figure()
plt.subplot(221)
plt.imshow(s)
plt.subplot(222)
plt.imshow(E)
plt.subplot(223)
plt.imshow(filtx)
plt.subplot(224)
plt.imshow(filty)
plt.show()

aes_ = np.sqrt((E*((filtx**2)+(filty**2))).sum()) / E.sum()
print(aes_)

# ------------------------------- #
"""
sed 's/\[/ /g' results_iqa_T1_OFF.txt > results_iqa_T1_OFF-test.txt

sed 's/\*/t/g' test.txt > test2.txt

The easiest way is to use sed (or perl):

sed -i -e 's/abc/XYZ/g' /tmp/file.txt
Which will invoke sed to do an in-place edit due to the -i option. This can be called from bash.

If you really really want to use just bash, then the following can work:

while read a; do
    echo ${a//abc/XYZ}
done < /tmp/file.txt > /tmp/file.txt.t
mv /tmp/file.txt{.t,}
"""



"""
a = nib.load('./OFF/PD/au70_PD_OFF.nii.gz').get_fdata()
a = a/a.max()
max_ = np.asarray(a.shape).max()
s = a[:,:,0]
s = s/s.max()
print(s.min(), s.max())

distances = [3, 5, 7, 9, 11, 21]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
properties = ['energy', 'homogeneity']

tmp_ =(np.uint8(s*255))

print(tmp_.min(), tmp_.max())

glcm = greycomatrix(tmp_, 
                    distances=distances, 
                    angles=angles,
                    symmetric=False,
                    normed=True)
print(glcm.shape)

tmp= glcm.sum(axis=2).sum(axis=2)
print(tmp.shape, shannon_entropy(tmp/tmp.max()))



plt.figure()
plt.subplot(131)
plt.imshow(s, cmap='gray')
plt.subplot(132)
plt.imshow(tmp_, cmap='gray')
plt.subplot(133)
plt.imshow(tmp, vmin=0.0002, vmax=0.003)

plt.show()
"""