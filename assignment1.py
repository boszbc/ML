# -*- coding: utf-8 -*-
""" Camera Obscura - Post-processing
This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

Notes
-----
You are only allowed to use cv2.imread, c2.imwrite and cv2.copyMakeBorder from 
cv2 library. You should implement convolution on your own.
GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but these functions should NOT save the image to disk.
    2. DO NOT import any other libraries aside from those that we provide.
    You should be able to complete the assignment with the given libraries
    (and in many cases without them).
    3. DO NOT change the format of this file. You may NOT change function
    type signatures (not even named parameters with defaults). You may add
    additional code to this file at your discretion, however it is your
    responsibility to ensure that the autograder accepts your submission.
    4. This file has only been tested in the course virtual environment.
    You are responsible for ensuring that your code executes properly in the
    virtual machine environment, and that any changes you make outside the
    areas annotated for student code do not impact your performance on the
    autograder system.
"""
import numpy as np
import cv2

#path = 'C:\\Users\\Binchi\\Desktop\\6475\\Raw Pictures\\co_image_0.jpg'
#img = cv2.imread(path,0)
#filter = 0.1 * np.ones([3,3])

def applyConvolution(image, filter):
    """Apply convolution operation on image with the filter provided. 
    Pad the image with cv2.copyMakeBorder and cv2.BORDER_REPLICATE to get an output image of the right size
    Parameters
    ----------
    image : numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8
    filter: numpy.ndarray
        A numpy array of dimensions (N,M) and type np.float64
    Returns
    -------
    output : numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8
    """
    
    # WRITE YOUR CODE HERE.

    nrows = image.shape[0]
    ncols = image.shape[1]
    out = np.zeros([nrows,ncols])
    
    for i in range(1,nrows-1):
        for j in range(1, ncols-1):
            temp_array = image[i-1:i+2,j-1:j+2]
            product = np.multiply(temp_array,filter)
            out[i,j] = int(np.sum(product))
            
    return out        
            
    raise NotImplementedError

def applyMedianFilter(image, filterdimensions):
    """Apply median filter on image after padding it with zeros around the edges using cv2.copyMakeBorder
    Parameters
    ----------
    image : numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8
    filterdimensions: list<int>
        List of length 2 that represents the filter size M x N
    Returns
    -------
    output : numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8
    """
    # WRITE YOUR CODE HERE.
    M, N = filterdimensions
    nrows = image.shape[0]
    ncols = image.shape[1]
    out = np.zeros([nrows,ncols])
    
    for i in range(int((M-1)/2),int(nrows-(M-1)/2)):
        for j in range(int((M-1)/2), int(ncols-(M-1)/2)):
            temp_array = image[int(i-(M-1)/2):int(i+(M-1)/2)+1,int(j-(M-1)/2):int(j+(M-1)/2)+1]           
            out[i,j] = np.sum(np.median(temp_array))
            
    return out        
    

    
    
    raise NotImplementedError

def applyFilter1(image):
    """Filter noise from the image by using applyConvolution() and an averaging filter
    Parameters
    ----------
    image : numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8
    
    Returns
    -------
    output : numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8
    """
    # WRITE YOUR CODE HERE.
    avg_filter = 1/9 * np.ones([3,3])
    return applyConvolution(image,avg_filter)

    raise NotImplementedError

def applyFilter2(image):
    """Filter noise from the image by using applyConvolution() and a gaussian filter
    Parameters
    ----------
    image : numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8
    
    Returns
    -------
    output : numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8
    """
    # WRITE YOUR CODE HERE.
    g_f = np.ones([3,3])
    sigma = 1
    for i in range(3):
        for j in range(3):
            u = i - 1
            v = j - 1
            index = - (u**2 + v**2)/ 2*sigma*sigma
            g_f[i,j] = 1/ (2 * np.pi * np.power(sigma,2)) * np.exp(index)
    
    gaussian = applyConvolution(image,g_f)
    return gaussian
    raise NotImplementedError
    
def sharpenImage(image):
    """Sharpen the image. Call applyConvolution with an image sharpening kernel
    Parameters
    ----------
    image : numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8
    
    Returns
    -------
    output : numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8
    """
    # WRITE YOUR CODE HERE.
    s_k = np.zeros([3,3])
    s_k[1,1] = 2
    s_k = s_k - 1/9 * np.ones([3,3])
    
    return applyConvolution(image,s_k)
    raise NotImplementedError

if __name__ == "__main__":
    # WRITE YOUR CODE HERE.
    # Read co_image_0.jpg and pass it to applyFilter1(), applyFilter2(), applyMedianFilter() and sharpenImage()
     image = cv2.imread("co_image_0.jpg",0)
     applyFilter1(image)
     applyFilter2(image)
     applyMedianFilter(image,(3,3))
     sharpenImage(image)
    pass


