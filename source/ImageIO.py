"""imageIO defines function for reading and writing images in different format"""

import os
from PIL import Image
import numpy as np
import TImage
from scipy import misc
import tiffLib

def imReader(path, fname, imformat, dim = 3):
    """ this function has three input arguments:
        path:      the path of the image to be read
        name:      the name of the image to be read(including extension)
        imformat:  the format of the image"""
          
    # check file existence
    if not os.path.isdir(path):
        print 'Directory does not exist!'
        return
    if not os.path.isfile(path + fname):
        print 'File does not exist!'
        return
    
    imfile = open(path + fname,'r')
    
    # read files in all formats
    if imformat == 'smv':
        '''SMV Format'''
        smvHeaderSize = 512       
        return readSMV(imfile,smvHeaderSize)
    
    if imformat == 'tif':
        '''TIFF Format'''
        return readTiff(path + fname,dim)
    
def readSMV(imfile,headerSize):
    """ this function reads a smv file into a stack of raw images """

    ## read header
    header = imfile.read(headerSize)
    splitheader = header.split('\n')

    items = []
    for i in splitheader:
        if i.find('=') != -1:
            items.append(i)

    for item in items:
        item = item.split('=')
        # Dimension
        if item[0].find('DIM') != -1:
            dim = item[1].strip().strip(';')
            dim = int( dim )
            print dim
        # Width
        if item[0].find('SIZE1') != -1:
            size_0 = item[1].strip().strip(';')
            size_0 = int( size_0)
            print size_0
        # Height
        if item[0].find('SIZE2') != -1:
            size_1 = item[1].strip().strip(';')
            size_1 = int( size_1)
            print size_1
        # Depth
        if item[0].find('SIZE3') != -1:
            size_2 = item[1].strip().strip(';')
            size_2 = int( size_2 )
            print size_2
        # Data Type
        if item[0].find('TYPE') != -1:
            data_type = item[1].strip().strip(';')
            print data_type

    smvIm = TImage.TImage();
    smvIm.setDim(dim)
    smvIm.setSize(size_0,size_1,size_2)
    smvIm.setDataType(data_type)
          
    ## read data                       
    data = []
    data_type = 2
    for l in range(size_2):
        chunk = imfile.read(size_0*size_1*data_type)
        im_array = np.frombuffer(chunk, dtype=np.uint16)
        im_array.resize(size_1, size_0)      
        #im = Image.fromarray(np.uint16(im_array),'L')
        data.append(im_array)

    smvIm.setData(data)
    
    return smvIm

def readTiff(fname,dim):
    
    tifIm = TImage.TImage();
    tifIm.setDim(dim)
    
    if dim>2:
        im_array = tiffLib.imread(fname)
        im_shape = im_array.shape
        
        tifIm.setSize(im_shape[1],im_shape[2],im_shape[0])
        data = []
        for i in range(im_shape[0]):
            data.append(im_array[i])
        tifIm.setData(data)
    else:
        im_array = tiffLib.imread(fname)
        im_shape = im_array.shape
        
        tifIm.setSize(im_shape[0],im_shape[1],1)
        data = []
        data.append(im_array)
        tifIm.setData(data)

    return tifIm
        
def imWriter(path, fname, im, dim = 3):
    """ this function has four input arguments:
        path:      the path of the image to be write
        name:      the name of the image to be write (including extension)
        im:        actual image object to be write
        dim:       output image dimensionality"""

    # check file existence
    if not os.path.isdir(path):
        print 'Directory does not exist, requested directory created!'
        os.makedirs(path)
        
    if  os.path.isfile(path + fname):
        print 'File exists, to be overwritten!'
        os.remove(path + fname)

    # save image as slices
    if dim == 2:
        fname = fname.split('.')
        for i in range(im.size_2):
            outputName = path + fname[0] + '_' + str(i) + '.tif'
            #misc.imsave(outputName,im.data[i])
            #im.data[i].save(outputName, fname[1])
            tiffLib.imsave(outputName,im.data[i])

    # save images as a stack
    if dim == 3:
        im_array = np.asarray(im.data)
        fname = fname.split('.')
        outputName = path + fname[0] + '.tif'     
        tiffLib.imsave(outputName,im_array)
            
        
