"""This script shows how to run the image format converting from smv to tiff(3D).
The scipt only runs in linux system due to windows python buffering problem."""

import sys, os, re, operator, pdb, subprocess, time

from scipy import misc
import numpy
import tiffLib

import ImageIO
import TImage
import ShapeIndex
import histEqualization
import AT_denoising


def test_func():
    """You can conver a single smv file to 3D tiff by specifying
    data_path, output_path, fileName and set the SigleConvert flag to 1.

    You can conver a batch of smv files to 3D tiffs by specifying
    root data_path, output_path and set the BatchConvert flag to 1.

    """

    ## Please specify paths ##
    data_path = '/home/yanbin/Tomosynthesis/data/SAP_test_datasets/Screening_30_cases/6002/'
    output_path = '/home/yanbin/Tomosynthesis/script_test/'
    fileName = '6002L06.smv'

    ## Please specify Parameters ##
    BatchConvert = 0
    SigleConvert = 0
    dim = 3

    ## data_path check
    if not os.path.isdir(data_path):
	print "Data directory:\n"+ data_path +"\ndoes not exist"
	sys.exit()

    ## Format convert batch
    if BatchConvert == 1:
        print 'here'
        dir_list = os.listdir(data_path)
        print dir_list
        for dirc in dir_list:
            print dirc
            if os.path.isdir(data_path + dirc):
                # make directory for output files
                opath = output_path + dirc + '/'
                print opath
                if not os.path.isdir(opath):
                    os.makedirs(opath)
                    
                file_list = os.listdir(data_path + dirc)    
                for fil in file_list: 
                    im = ImageIO.imReader(data_path + dirc + '/', fil, 'smv')
                    ImageIO.imWriter(opath, fil.strip('smv') + 'tif',im,dim)

    ## Format convert single
    if SigleConvert == 1:
        im = ImageIO.imReader(data_path,fileName, 'smv')
        ImageIO.imWriter(output_path, fileName.strip('smv') + 'tif',im, dim)


if __name__ == '__main__':

    test_func()                 
