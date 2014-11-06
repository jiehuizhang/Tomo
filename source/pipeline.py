import sys, os, re, operator, pdb, subprocess, time

from scipy import misc
import numpy
import tiffLib

import ImageIO
import TImage
import ShapeIndex
import histEqualization
import AT_denoising

def main():

    ## Please specify paths ##
    data_path = '/home/yanbin/Tomosynthesis/data/SAP_test_datasets/Screening_30_cases/'
    output_path = '/home/yanbin/Tomosynthesis/data/tiffs_3d/'
    exe_path= '/home/yanbin/Tomosynthesis/code/'

    ## Please specify Run Flags ##
    FormatConvert = 1		
    AWDenoising = 0
    ContrastEnhancement = 0

    ## Please specify parameters ##
    dim = 3               # For format convert: save as 2d slices / 3d stack
    opt = 'asymptotic'    # For AWdenoising inverse transform options
    block_m=5
    block_n=5   # For AWdenoising Wiener filter window size block_m = block_n

    ###################### Avalability Check #######################

    # data_path check
    if not os.path.isdir(data_path):
	print "Data directory:\n"+ data_path +"\ndoes not exist"
	sys.exit()
	
    # exe_path check
    if not os.path.isdir(exe_path):
	print "Executable directory:\n"+ exe_path +"\ndoes not exist"
	sys.exit()
  
    ###################### Format Convert #######################
    
    if FormatConvert == 1:
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

    ############################# Denoising ##########################
    if AWDenoising == 1:
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
                    im = ImageIO.imReader(data_path + dirc + '/', fil, 'tif',2)
                    denoised = AT_denoising.DenoisingAW(im.data[0], opt = 'asymptotic', block_m=5,block_n=5)
                    tiffLib.imsave(opath + 'denoised_' + fil,denoised)


    ###################### Contrast enhancement #######################
    if ContrastEnhancement == 1:
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
                    im = ImageIO.imReader(data_path + dirc + '/', fil, 'tif',2)
                    enhanced = histEqualization.histEqualization(im.data[0], 16)
                    tiffLib.imsave(opath + 'enhanced_' + fil,enhanced)
	    
main()    




    
