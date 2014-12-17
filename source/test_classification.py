"""This script shows how to run creating training samples and classification.
There is no binary classification included (performance is not good as reported,
example of classification by multi instance learning is included in the script. )
"""

import pickle
import Creat_trainSam
import midt

import numpy as np

def test_func():
    """Set CreatTraining to 1 and specify cancerPath/controlPath if you want to
    run create training samples.You should have training image cropps (3D) within
    the directory.
    Set MIL to 1 if you want to run classification. all wrokspace files should be in
    the outputPath.    
    """

    cancerPath = 'C:/Tomosynthesis/training/cancer_3d/'
    controlPath = 'C:/Tomosynthesis/training/control_3d/'
    outputPath = 'C:/Tomosynthesis/test_script/'

    ## running flags
    CreatTraining = 0
    MIL = 1

    ## Creat training samples
    if CreatTraining == 1:
        
        LightPatchList = Creat_trainSam.creatTrainigSam_3D(cancerPath)
        output = open(outputPath + 'cancer.pkl', 'wb')
        pickle.dump(LightPatchList, output)
        output.close()

        LightPatchList = Creat_trainSam.creatTrainigSam_3D(controlPath)
        output = open(outputPath + 'control.pkl', 'wb')
        pickle.dump(LightPatchList, output)
        output.close()
    
    
    ## run mil classification
    if MIL == 1:
        sus_file = open(outputPath + 'suspicious.pkl', 'rb')
        sliceList = pickle.load(sus_file)
        sus_file.close()
        
        cancer_file = open(outputPath + 'cancer.pkl', 'rb')
        cancerList = pickle.load(cancer_file)
        cancer_file.close()
        
        control_file = open(outputPath + 'control.pkl', 'rb')
        controlList = pickle.load(control_file)

        print 'classifying ...'
        midt.classify(sliceList, cancerList, controlList)


if __name__ == '__main__':

    test_func()
