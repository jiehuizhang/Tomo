"""Multi Instance Desiicion Tree Classfication """

import numpy as np
from sklearn import tree

import ImageIO
import TImage
import TPatch

import Dimreduction

def ClusteringtoBags(coord, dis_thresh = 15):
    """Clustering ROIs in slices into 3D ROIs, so that 2D ROI with close (x,y)
    coordination from adjacent slices  will be assigned into one bag.

    Parameters
    ----------
    coord: list of coordination
        The coordination list for all ROIs
    dis_thresh:
        Threshold for assigning ROIs into the same bag.
    
    """

    bagIDS = []
    bagIDS.append(0)
    global_bid = 1
    for i in range(1,len(coord)):
        center_i = coord[i]
        for j in range(i):
            center_j = coord[j]

            dis = np.sqrt(np.sum((np.asarray(center_i) - np.asarray(center_j))**2))
            if dis < dis_thresh:
                bagIDS.append(bagIDS[j])
                break           

            if j == i - 1:
                bagIDS.append(global_bid)
                global_bid = global_bid + 1
    return (bagIDS,global_bid)

def info_fetch(plist,opt):
    """ Fetch features information from the list

    Parameters
    ----------
    plist: list of the data set
        The list include ROIs and all information within
    opt: atr
        If the required information are for taining or testing
    
    """

    coord = []
    bagIDS = []
    feats = None
    if opt == 'test':
        for i in range(len(plist)):

            if feats == None:
                feats = plist[i].feats
            else:
                if plist[i].feats == None:
                    plist[i].feats = np.zeros((1,feats.shape[1]))
                    print "Nan feature occured!"
                feats = np.vstack((feats, plist[i].feats))
                               
            for j in range(len(plist[i].LightPatchList)):
                coord.append(plist[i].LightPatchList[j].image_center)
                
        return (feats, coord)

    if opt == 'train':
        for i in range(len(plist)):
            if feats == None:
                feats = plist[i].feats
            else:                
                feats = np.vstack((feats, plist[i].feats))
                
            bagIDS.append(plist[i].bagID)

        return (feats,bagIDS)
    
def classify(sliceList, cancerList, controlList):
    """Main function for classification using multi instance desicion tree

    Parameters
    ----------
    sliceList:
        List of suspicious data
    cancerList:
        List positive data
    controlList:
        List of negative data
    """

    # Fetch feature and coordinate information from list
    fsus,coordsus = info_fetch(sliceList, opt = 'test')
    fcancer,bid_canc = info_fetch(cancerList, opt = 'train')
    fcontrol,bid_cont = info_fetch(controlList, opt = 'train')

    numcanc = fcancer.shape[0]
    numcont = fcontrol.shape[0]
    numsus = fsus.shape[0]

    # clustering suspicious to bags of ROI
    bid_sus,bsize = ClusteringtoBags(coordsus)

    # feature normalization
    
    # dimension reduction
    false_lab = np.zeros((numcanc+numcont+numsus,0))
    data_projected = Dimreduction.dim_Reduction(np.vstack((fcancer,fcontrol,fsus)), false_lab, opt ='spectral',
                                                        n_components=5, visualize = False)    
    # training desicion tree
    clc = tree.DecisionTreeClassifier()
    clc.fit(data_projected[:numcanc+numcont,:],np.hstack( (np.ones(numcanc),np.zeros(numcont) ) ) )
    
    # classification instances
    predicts = clc.predict(data_projected[numcanc+numcont:,:])

    # assigning labels for each bag
    score = []
    for i in range(bsize):
        mask = np.asarray(bid_sus) == i
        score.append( np.sum(predicts[mask])/predicts[mask].size )
        if score[i]>0.5:
            print (i, score[i], coordsus[np.where(np.asarray(bid_sus) == i)[0][0]] )



     

    

