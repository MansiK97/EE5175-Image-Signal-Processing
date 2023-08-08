#!/usr/bin/env python
# coding: utf-8

# # LAB ASSIGNMENT 07 OF IMAGE SIGNAL PROCESSING
# 
# ## DFT, Magnitude-Phase Dominance, and Rotation Property
# 
# **MANSI KAKKAR**
# 
# **EE21S063**
# 

# **IMPORTING LIBRARIES**

# In[1]:


import imageio as io
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.fft import fftshift, fft


# In[2]:


get_ipython().system('ls')


# **READING THE IMAGES**

# In[3]:


#reading fourier and fourier transform image
fourier = io.imread('fourier.png')
fourier_transform = io.imread('fourier_transform.png')


# In[4]:


#reading peppers image
peppers = io.imread('peppers_small.png')
plt.imshow(peppers,'gray')


# **DISPLAYING THE IMAGES**

# In[5]:


plt.figure(figsize = (6,6))
plt.imshow(fourier,cmap = 'gray')
plt.show()
plt.figure(figsize = (6,6))
plt.imshow(fourier_transform,cmap = 'gray')
plt.show()


# **FUNCTION FOR BILINEAR INTERPOLATION**

# In[6]:


#FUNCTION FOR BILINEAR INTERPOLATION

def bilinear_interpolation(source,x,y):
    
    #taking coordinates of zero padded image
    xz,yz=np.shape(source)
    #original coordinates of image
    xz=xz-2 
    yz=yz-2
    
    #taking x+1 and y+1 since coordinates have to be aligned to the zero padded image
    #zero padded image has one coordinate +1 due to padding hence target image coordinate is incremented by 1
    x=x+1
    y=y+1
    
    #defining x',y' as floor of x,y 
    x_prime=math.floor(x)
    y_prime=math.floor(y)
    #defining a,b as distance of target point from floor points
    a=x-x_prime
    b=y-y_prime
    
    if x_prime>=0 and x_prime<=xz and y_prime>=0 and y_prime<=yz:
        #calculating the intensity at the target
        intensity = (1-a)*(1-b)*source[x_prime,y_prime]+(1-a)*b*source[x_prime,y_prime+1]+a*(1-b)*source[x_prime+1,y_prime]+a*b*source[x_prime+1,y_prime+1]
        #converting intensity into integer as intensity value cannot be float
        intensity=int(intensity)
    else:
        #when xs and ys are not defined i.e. xs and ys are <0 for corresponding target location
        intensity = 0

    return intensity


# **FUNCTION FOR ROTATION**

# In[7]:


#FUNCTION FOR ROTATION

def rotation(source_image, theta):
  
  #reading image size
  x , y = np.shape(source_image)

  #creating a blank image
  rot_image=np.zeros((x,y))

  #zero padded source image
  zero_padded_image=np.zeros((x+2,y+2))
  zero_padded_image[1:-1,1:-1]=source_image

  #calculating center coordinates sicne rotation of the image has to be done about the center of the image
  x_center=x/2  
  y_center=y/2

  #theta : degrees to radians
  theta=theta*(np.pi)/180

  #nested for loops for target coorinates ranging to source coordinates
  for xt in range(x):
    for yt in range(y):
      #every target point to be rotated about the center
      xt_c=xt-x_center
      yt_c=yt-y_center

      #applying clockwise rotation matrix :
      #   [Xt] = [clockwise rotation matrix][Xs]
      #   Considering inverse rotation matrix
      xs = np.cos(theta)*xt_c - np.sin(theta)*yt_c + x_center
      ys = np.sin(theta)*xt_c + np.cos(theta)*yt_c + y_center
      intensity=bilinear_interpolation(zero_padded_image,xs,ys)
      rot_image[xt,yt]=intensity
      
  return rot_image


# **2D DFT FUNCTION**
# 
# Calculating 2D DFT of image. Use the separability property of 2D DFT to break it into row and column transforms. The row and column transforms can be performed using a 1D DFT.

# In[8]:


def dft_2d(source):
    #getting shape of source
    x,y = np.shape(source)
   
    #row transform
    transform_r = fft(source, axis=1)
    #transform on row transformed image
    dft_transform = fft(transform_r, axis=0)
    #fftshift to the center for more intuitive visualization
    dft_transform = fftshift(dft_transform)
    
    # getting the magnitude and phase
    mag = np.abs(dft_transform)
    phase = np.zeros_like(dft_transform)
    phase[mag!=0] = dft_transform[mag!=0]/mag[mag!=0]

    return mag, phase


# **IDFT FUNCTION**
# 
# Calculate the 2D IDFT for the given image. Implementation is very similar to the dft_2D function. Use the property that taking DFT of the DFT of an image gives a flipped image to compute the IDFT.

# In[9]:


def idft_2d(mag, phase, shift_first = True):
    #get dft from given maginitude and phase 
    dft = mag*phase
    #getting shape of dft
    x,y = np.shape(dft)
   
    if shift_first:
        dft = fftshift(dft)
    #row transform
    transform_r = fft(dft, axis=1)
    #transform on row transformed image
    transform = fft(transform_r, axis=0)
    #fftshift to the center for more intuitive visualization
    if not shift_first:
        transform = fftshift(transform)
    #magnitude 
    mag = np.abs(transform)
    # flipping the image 
    mag = mag[::-1, ::-1]
    
    return mag


# **FUNCTION FOR PHASE SWAPPING AND OBSERVING THE OUTPUT**
# 
# Checking for the effect of swapping the phases of the images, and seeing what effect the phases cause on the image. As given in the question $F_{1}(k,l) = |F_{1}(k, l)|e^{j\Phi_{1}(k, l)}$ and $F_{2}(k,l) = |F_{2}(k, l)|e^{j\Phi_{2}(k, l)}$ are DFTs of I1 and I2 respectively. 
# 
# We have to create two new images such that I3 and I4 are $F_{1}(k,l) = |F_{1}(k, l)|e^{j\Phi_{2}(k, l)}$ and $F_{2}(k,l) = |F_{2}(k, l)|e^{j\Phi_{1}(k, l)}$ respectively

# In[10]:


def phase_swap(img1, img2):
    #get DFTs of both the images
    m1, p1 = dft_2d(img1)
    m2, p2 = dft_2d(img2)
    #getting the swapped phases images
    img12 = idft_2d(m1, p2)
    img21 = idft_2d(m2, p1)
    return img12, img21


# **ROTATION USING DFT**
# 
# Compute rotated form of 2D DFT such that the image center is the origin as:
# 
# 
# $F(k,l) = \sum_m \sum_n f(m,n)\exp{-j\frac{2\pi}{N}{m'}^TR{k'}}$ 
# 
# where 
# 
# 
# ${m'}=[m n]^T$ 
# 
# and
# 
# ${k'}=\begin{bmatrix} \frac{k-\frac{M}{2}}{M} & \frac{l-\frac{N}{2}}{N} \end{bmatrix}^T$
# 
# 
# and Rotation Matrix is 
# 
# $R = \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}$
# 

# In[11]:


def rotated_dft(image, theta):
    
    theta *= np.pi/180
    #Rotation matrix
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    
    # get the shape of the image
    x, y = np.shape(image)
    
    # make a canvas of zeroes ensure that the data type is complex
    F = np.zeros((x, y), dtype=np.complex)
    
    # use meshgrid for vectorized operations
    X, Y = np.meshgrid(np.arange(x), np.arange(y))
    
    # get the image center
    x_c = x//2
    y_c = y//2
    
    #reshaping
    m_vec = np.hstack((X.reshape(-1, 1)-x_c, Y.reshape(-1, 1)-y_c))
    for i in range(x):
        for j in range(y):
            # get the k vector as mentioned
            k_vec = np.array([(i-x_c)/x, (j-y_c)/y])
            k_vec = (rot_mat@k_vec)
            vals = m_vec@k_vec
            # performing vectorized multiplication and reshaping
            vals = (vals.reshape(y, x)).T
            phase = np.exp(-2j*np.pi*(vals))
            # assign the required value
            F[i, j] = np.sum(image*phase)
    #magnitude and phase
    mag = np.abs(F)
    phase = F/mag
    
    return mag, phase


# # RESULTS

# **DFT OF GIVEN IMAGES**

# In[12]:


m_f, p_f = dft_2d(fourier)
m_ft, p_ft = dft_2d(fourier_transform)
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6))=plt.subplots(2,3,figsize=(12,12),constrained_layout=True)
ax1.imshow(fourier,'gray')
ax1.set_title('fourier image')
ax2.imshow(np.log10(m_f),'gray')
ax2.set_title('magnitude of fourier image')
ax3.imshow(np.angle(p_f),'gray')
ax3.set_title('phase of fourier image')
ax4.imshow(fourier_transform,'gray')
ax4.set_title('fourier tansform image')
ax5.imshow(np.log10(m_ft),'gray')
ax5.set_title('magnitude of fourier tansform image')
ax6.imshow(np.angle(p_ft),'gray')
ax6.set_title('phase of fourier transform image')


# **PHASE SWAPPING RESULT**

# In[13]:


m1_p2, m2_p1 = phase_swap(fourier, fourier_transform)
fig, ((ax1,ax2))=plt.subplots(1,2,figsize=(9,9),constrained_layout=True)
ax1.imshow(m1_p2,'gray')
ax1.set_title(r"$F(k,l) = \|F_{1}(k, l)\|e^{j\Phi_{2}(k, l)}$")
ax2.imshow(m2_p1,'gray')
ax2.set_title(r"$F(k,l) = \|F_{2}(k, l)\|e^{j\Phi_{1}(k, l)}$")
plt.show()


# **ROTATION PROPERTY OF 2D DFT**

# In[14]:


theta = 30
peppers_rotated = rotation(peppers, theta)
m, p = rotated_dft(peppers, theta)
peppers_rotated_dft = idft_2d(m, p, shift_first=False)
fig, ((ax1,ax2))=plt.subplots(1,2,figsize=(9,9),constrained_layout=True)
ax1.imshow(peppers_rotated,'gray')
ax1.set_title('peppers_rotated using normal rotate function')
ax2.imshow(peppers_rotated_dft,'gray')
ax2.set_title('peppers_rotated using DFT')


# In[15]:


m_r, p_r = dft_2d(peppers_rotated)

fig, ((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(9,9),constrained_layout=True)
ax1.imshow(np.log10(m_r),'gray')
ax1.set_title('magnitude of peppers_rotated using normal rotate function')
ax2.imshow(np.log10(m),'gray')
ax2.set_title('magnitude of peppers_rotated using DFT')
ax3.imshow(np.angle(p_r),'gray')
ax3.set_title('phase of peppers_rotated using normal rotate function')
ax4.imshow(np.angle(p),'gray')
ax4.set_title('phase of peppers_rotated using DFT')


# # OBSERVATIONS:
# 
# * Maximum energy of image is concentrated at the center of the image
# * Phase carries maximum information of the image, since we can observe in our comparison by swapping the phases in the images
# * The DFT magnitude of fourier_transform image contains a distinct horizontal line which is not present in fourier image, we can claim that the former contains a lot of vertical edges in the spatial domain, which give rise to a horizontal line in the DFT
# * Rotating the image in spatial domain or rotating the image using DFT give us identical results. We can observe very slight changes in magnitude and phase plot of both which can be ignored
