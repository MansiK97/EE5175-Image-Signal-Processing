import imageio
import math
import numpy as np
from matplotlib import pyplot as plt

#%%
#reading image
lena=imageio.imread("lena_translate.png")
pisa=imageio.imread("pisa_rotate.png")
cells=imageio.imread("cells_scale.png")

#%%
#TRANSLATION
def translate(source_image,tx,ty):
    x,y=np.shape(source_image)
    #creating blank image
    trans_image=np.zeros((x,y))
    #zero padded source image
    zero_padded_image=np.zeros((x+2,y+2))
    zero_padded_image[1:-1,1:-1]=source_image
    
    for xt in range(x):
        for yt in range(y):
            xs=xt-tx
            ys=yt=ty
            intensity=bilinear_interpolation(zero_padded_image,xs,ys)
            trans_image[xt,yt]=intensity
    return trans_image
    
#%%
#ROTATION
def rotation(source_image,theta):
    x,y=np.shape(source_image)
    #creating blank image
    rot_image=np.zeros((x,y))
    #zero padded source image
    zero_padded_image=np.zeros((x+2,y+2))
    zero_padded_image[1:-1,1:-1]=source_image
    x_center=x/2
    y_center=y/2
    theta=theta*(np.pi)/180
    
    for xt in range(x):
        for yt in range(y):
            xt_c=xt-x_center
            yt_c=yt-y_center
            xs=np.cos(theta)*xt_c-np.sin(theta)*yt_c
            ys=np.cos(theta)*xt_c+np.sin(theta)*yt_c
            intensity=bilinear_interpolation(zero_padded_image,xs,ys)
            rot_image[xt,yt]=intensity
    return rot_image

#%%
#SCALING
def scaling(source_image,a):
    x,y=np.shape(source_image)
    #creating blank image
    scale_image=np.zeros((x,y))
    #zero padded source image
    zero_padded_image=np.zeros((x+2,y+2))
    zero_padded_image[1:-1,1:-1]=source_image
    if a>0:
        for xt in range(x): 
            for yt in range(y):
                xs=xt/a
                ys=yt/a
                intensity=bilinear_interpolation(zero_padded_image,xs,ys)
                scale_image[xt,yt]=intensity
    return scale_image
    
#%%
#BILINEAR INTERPOLATION
def bilinear_interpolation(source,x,y):
    x=x+1
    y=y+1
    x_prime=math.floor(x)
    y_prime=math.floor(y)
    a=x-x_prime
    b=y-y_prime
    xz,yz=np.shape(source)
    xz=xz-2
    yz=yz-2
    if x_prime>=0 and x_prime<=xz and y_prime>=0 and y_prime<=yz:
        intensity = (1-a)*(1-b)*source[x_prime,y_prime] \
            +(1-a)*b*source[x_prime,y_prime+1] \
            +a*(1-b)*source[x_prime+1,y_prime] \
            +a*b*source[x_prime+1,y_prime+1]
        print(intensity)
    else:
        intensity = 0
    return intensity
#%%
#calling functions
Tx=3.75
Ty=4.3
theta=4
a1=0.8 
a2=1.3
#new_lena=translate(lena, Tx, Ty)
new_pisa=rotation(pisa, theta)
new_cells_zoom_out=scaling(cells, a1)
new_cells_zoom_in=scaling(cells, a2)
fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8))=plt.subplots(4,2)
ax1.imshow(lena)
ax1.title('Original Image')
#ax2.imshow(new_lena)
#ax2.title('Translated Image by tx=3.75 and ty=4.3')
ax3.imshow(pisa)
ax3.title('Original Image')
ax4.imshow(new_pisa)
ax4.title('Rotated Image by angle 4')
ax5.imshow(cells)
ax5.title('Original Image')
ax6.imshow(new_cells_zoom_out)
ax6.title('Scaled Image by scaling factor = 0.8')
ax7.imshow(cells)
ax7.title('Original Image')
ax8.imshow(new_cells_zoom_in)
ax8.title('Scaled Image by scaling factor = 1.3')
plt.show()

#%%

    