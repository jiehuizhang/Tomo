"""This script shows how to run suspicious mass extraction start from raw
images. Preprocessing are included within the parallel process."""
import pickle
from multiprocessing import Pool

import ImageIO
import TImage
import mass3Dextraction as mass3d


def test_func():
    """Please specify the image tiff stack directory, file name and output
    path. Extracted masses list will be save in a workspace file for further use.
    Also please specify how many cores will be allocated for the parallel process.
    """


    dataPath = '/home/yanbin/Tomosynthesis/data/tiffs_3d/5016/'
    outputPath = '/home/yanbin/Tomosynthesis/script_test/'
    fileName = '5016EMML08.tif'

    ## load image data
    im = ImageIO.imReader(dataPath,fileName, 'tif')

    ## allocate cpu source
    pool = Pool(processes=6)
    params =[(i,im.data[i]) for i in range(im.size_2)]

    ## run in parallel
    sliceList = []    
    sliceList = pool.map(mass3d.parallelWrapper,params)
    
    ## save the workspace
    output = open(outputPath + 'suspicious.pkl', 'wb')
    pickle.dump(sliceList, output)
    output.close()


if __name__ == '__main__':

    test_func()
