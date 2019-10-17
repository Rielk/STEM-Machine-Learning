# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 19:48:48 2018

@author: theisw
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

np.seterr(all='ignore')

def intersect_heights(p_arr,tri):
    """ adapted from https://www.erikrotteveel.com/python/three-dimensional-ray-tracing-in-python/ 
    
    Parameters
    ==========
        p: arrays of poins, shape: 
    """
    v2, v0, v1 = tri
    u = v1 - v0
    v = v2 - v0
    normal = np.cross(u, v)
    d = np.array([[0,0,1]])
    b = np.inner(normal, d)     # b == 0 means no intersection
    a = np.inner(normal, p_arr - v0)
    h = -a/b    # h is height
    # determine whether is inside
    ustar = np.cross(u, normal)
    vstar = np.cross(v, normal)    
    c1 = np.inner((p_arr - v0 + (h*d.T).T), ustar)/np.inner(v, ustar)
    c2 = np.inner((p_arr - v0 + (h*d.T).T), vstar)/np.inner(u, vstar)
    h[(c1<0)+(c2<0)+(c1+c2>1)+(b==0)] = 0    # + serves as OR in this context
    return h
   
    

class Convex:
    """ defined by the inside of a surface made up of triangles 
    triangles is an array with shape (n,3,3), triang nr, corner nr, x/y/z
    """
    
    def __init__(self, triangles):
        self.triangles = triangles
        self.triangles_original = triangles.copy()
        
    def euler_rotate(self, theta_deg):
        """ rotates corners by euler angles
        """
        theta = np.deg2rad(np.array(theta_deg))
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(theta[0]), -np.sin(theta[0])],
                        [0, np.sin(theta[0]), np.cos(theta[0])]])
        R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                        [0, 1, 0],
                        [-np.sin(theta[1]), 0, np.cos(theta[1])]])
        R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                        [np.sin(theta[2]), np.cos(theta[2]), 0],
                        [0, 0, 1]])
        R = np.dot(R_z, np.dot( R_y, R_x))
        self.triangles = np.dot(self.triangles_original, R.T)
        
    def image(self, n, pix_size=1, noise=None, background=0, noisescale=0.01, gauss=None, dydt=0, dxdt=0):
        """ calculates thickness image """
        a = np.arange(0,n)-n//2
        x,y = np.meshgrid(a*pix_size,a*pix_size)
        if dxdt != 0 or dydt != 0:
            t = np.arange(x.size)
            p_arr = np.stack([x.flatten() + t*dxdt, y.flatten() + t*dydt, np.zeros(n*n)]).T
        else:    
            p_arr = np.stack([x.flatten(),y.flatten(),np.zeros(n*n)]).T
        th = self.thickness(p_arr) + background
        th = th.reshape(n,n)
        if gauss is not None:
            th = gaussian_filter(th, sigma=gauss)
        if noise is not None:
            th += noisescale * np.random.poisson(lam=noise*th)
        return th
        
    def thickness(self, p_arr):
        """ thickness along z direction """
        thickn = np.zeros((self.triangles.shape[0], p_arr.shape[0])) 
        for k, t in enumerate(self.triangles):
            thickn[k,:] = intersect_heights(p_arr,t)
        t = np.nanmax(thickn, axis=0)-np.nanmin(thickn, axis=0)
        return t


if __name__ == '__main__':
            
    shapes = np.array([np.array([[[0,0,0], [0,0,1], [0,1,0]],
                    [[0,0,0], [0,0,1], [1,0,0]],
                    [[0,0,0], [0,1,0], [1,0,0]],
                    [[0,0,1], [0,1,0], [1,0,0]]])-0.5+0.002,
    
            np.array([[[0,0,0], [1,0,0], [0,1,0]], # bottom
                     [[1,1,0], [1,0,0], [0,1,0]], 
                     [[0,0,1], [1,0,1], [0,1,1]], # top
                     [[1,1,1], [1,0,1], [0,1,1]],
                     [[0,0,0], [0,1,0], [0,0,1]], # permute ->
                     [[0,1,1], [0,1,0], [0,0,1]],
                     [[1,0,0], [1,1,0], [1,0,1]], # 
                     [[1,1,1], [1,1,0], [1,0,1]],
                     [[0,0,0], [0,0,1], [1,0,0]], # permute ->
                     [[1,0,1], [0,0,1], [1,0,0]],
                     [[0,1,0], [0,1,1], [1,1,0]], # 
                     [[1,1,1], [0,1,1], [1,1,0]]])-0.5+0.002])

    for shape in shapes:
        points = np.unique([shape[x,y,:] for x in range(shape.shape[0]) for y in range(shape.shape[1])], axis=0)
        volume = ConvexHull(points).volume
        shape /= 2*volume**(1/3)
                     
    n = 256 # scan points/pixels
    nr_cols, nr_rows = 2, 20
    images = nr_rows * nr_cols
    fig = plt.figure(figsize=(8, 80))
    # generate shape    
    ims = []
    for nr, r in enumerate(np.random.rand(images, 3)*360):
        t = Convex(np.random.choice(shapes))
        t.euler_rotate(r)
        im = t.image(n, pix_size=2/n, noise=3, gauss=1.5, background=10)
#        im = t.image(n, pix_size=2/n, noise=3, gauss=1.5, background=10, dxdt=2e-5)
        ims.append(im)
        print("Shape {} complete".format(nr+1))
    for nr, im in enumerate(ims):
        norm = plt.Normalize(np.min(ims), np.max(ims))
        cmap = plt.cm.gray
        image = cmap(norm(im))
        plt.imsave("test/Test_{}.png".format(nr), image)
        ax = fig.add_subplot(nr_rows, nr_cols, nr+1)
        ax.imshow(im, cmap='gray',norm=norm)