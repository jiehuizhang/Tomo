""" Binaray classification using graphcut """
import time
import numpy as np
import cmath
import math

import ImageIO
import TImage
import TPatch

from pygraph.classes.graph import graph
from pygraph.classes.digraph import digraph
from pygraph.algorithms.minmax import minimal_spanning_tree,\
shortest_path, heuristic_search, shortest_path_bellman_ford, maximum_flow, cut_tree

from pyentropy import DiscreteSystem
import MI
import Dimreduction
import histEqualization
    
def computeFeatureCorrelation(featSuspicious, featSource, featSink):
    """Computer extracted feature correlation

    Parameters
    ----------
    featSuspicious:
        Testing feature table
    featSource:
        Training positive feature table
    featSink:
        Training negative feature table   

    """

    numSus = featSuspicious.shape[0]
    numSor = featSource.shape[0]
    numSin = featSink.shape[0]

    dataCombo = np.vstack((featSource, featSink, featSuspicious))
    
    # dimensionality reduction 
    n_components = 5
    data_projected = Dimreduction.dim_Reduction(dataCombo, None, 'pca', n_components)

    # Normialize feature vectors
    norm = normalize(data_projected)
    featSource = norm[0:numSor,:]
    featSink = norm[numSor:numSor+numSin,:]
    featSuspicious = norm[numSor+numSin:numSor+numSin+numSus,:]

    # for debugging
    #outputPath = 'C:/Tomosynthesis/localtest/res/'   
    #np.savetxt(outputPath + 'norm.txt', norm, delimiter='\t')
    
    weightFeat = np.zeros((numSus + numSor + numSin,numSus + numSor + numSin),dtype=np.double)
	
    val_min = 0.0
    val_max = np.sqrt(n_components)

    for i in range(numSor):
        for j in range(i,numSor):
            weightFeat[i][j] = val_min #computeDis(featSource[i],featSource[j])
        for j in range(numSin):
            weightFeat[i][j+numSor] = val_max #computeDis(featSource[i],featSink[j])
        for j in range(numSus):
            weightFeat[i][j+numSor+numSin] = computeDis(featSource[i],featSuspicious[j])

    for i in range(numSin):
        for j in range(i,numSin):
            weightFeat[i+numSor][j+numSor] = val_min #computeDis(featSink[i],featSink[j])
        for j in range(numSus):
            weightFeat[i+numSor][j+numSor+numSin] = computeDis(featSink[i],featSuspicious[j])

    for i in range(numSus):
        for j in range(i,numSus):
            weightFeat[i+numSor+numSin][j+numSor+numSin] = computeDis(featSuspicious[i],featSuspicious[j])			
        
    return weightFeat

def normalize(comb):
    """Feature normalization"""

    cmax = np.amax(comb, axis=0)
    cmin = np.amin(comb, axis=0)
    norm = (comb - cmin)/(cmax - cmin)
    return norm
    
def computeDis(feat1,feat2,opt = 'Euc'):
    """Compute distance of two vectors."""

    dis = 0.0
    if opt == 'Euc':
        """Euclidean distance"""    
        for i in range(feat1.shape[0]):
            dis = dis + (feat1[i] - feat2[i])**2
        dis = np.sqrt(dis)
        
    return dis

def padding(data1,data2):
    """Padding two images to the larger size"""

    m1,n1 = data1.shape
    m2,n2 = data2.shape

    if m1 > m2:
        up = (m1 - m2)/2
        down = m1 - m2 - up
        data2 = np.lib.pad(data2, ((up,down),(0,0)),'edge')
    if m1 < m2:
        up = (m2 - m1)/2
        down = m2 - m1 - up
        data1 = np.lib.pad(data1, ((up,down),(0,0)),'edge')
    if n1 > n2:
        left = (n1-n2)/2
        right = n1-n2 - left
        data2 = np.lib.pad(data2, ((0,0),(left,right)),'edge')
    if n1 < n2:
        left = (n2-n1)/2
        right = n2-n1 - left
        data1 = np.lib.pad(data1, ((0,0),(left,right)),'edge')

    return (data1,data2)

def _MI(pdfs1, pdfs2):
    """Compute mutual information of two probability density functions"""

    numR = len(pdfs1)
    mi_map = np.zeros(numR)
    
    # compute mi
    for i in range(numR):
      
        mi_map[i] = MI.MI(pdfs1[i],pdfs2[i])

    return np.sum(mi_map)


def getPdf(data, center, numR = 10,eq = False):
    """Compute the pdf functions of each rings from the ROI"""

    # equalization before calculating of pdf
    
    pdfs = []
    bins = np.arange(0, 1, 0.05)
    for k in range(len(data)):
        data[k] = histEqualization.histEqualization(data[k], 16)
        pdf = []
               
        r1 = 0
        dr = data[0].shape[0]/(2*numR)
        for i in range(numR):
            r2 = r1 + dr
            
            nr,nc = data[k].shape
            X, Y = np.ogrid[0:nr-1, 0:nc-1]
            mask1 = (X - center[k][0])**2 + (Y - center[k][1])**2 < r1**2
            mask2 = (X - center[k][0])**2 + (Y - center[k][1])**2 < r2**2
            mask = mask2-mask1
            ring = data[k][mask]      

            # normalize
            ring = np.double(ring - np.min(ring))/np.double(np.max(ring) - np.min(ring))
            pdfi= np.histogram(ring, bins)

            pdfi = pdfi[0]/np.double(np.sum(pdfi[0]))
            pdf.append(pdfi)
            

            r1 = r1 + dr
            dr = max(dr - 2,5)

        pdfs.append(pdf)

    return pdfs

   
def computeIntensityCorrelation(intenSuspicious, intenSource, intenSink, cenSuspicious, cenSource, cenSink):
    """Computer intensity correlation of two ROIs

    Parameters
    ----------
    intenSuspicious: list of numpy array(2D)
        Testing ROI intensity lists
    intenSource:
        Training positive ROI intensity lists
    intenSink:
        Training negative ROI intensity lists
    cenSuspicious: list of tuples with coordination
        The center coordinations of all suspicious data point
    cenSource: list of tuples with coordination
        The center coordinations of all positive data point
    cenSink: list of tuples with coordination
        The center coordinations of all negative data point
    
    """

    numSus = len(intenSuspicious)
    numSor = len(intenSource)
    numSin = len(intenSink)

    pdfsource = getPdf(intenSource, cenSource,eq = True)
    pdfsink = getPdf(intenSink, cenSink,eq = True)
    pdfSuspicious = getPdf(intenSuspicious, cenSuspicious,eq = False)

    weightIntens = np.zeros((numSus + numSor + numSin,numSus + numSor + numSin),dtype=np.double)
	
    val_min = 0.0
    val_max = 10.0

    for i in range(numSor):
        for j in range(i,numSor):
            weightIntens[i][j] = val_min #_MI(pdfsource[i],pdfsource[j])
        for j in range(numSin):
            weightIntens[i][j+numSor] = val_max #_MI(pdfsource[i],pdfsink[j])
        for j in range(numSus):
            weightIntens[i][j+numSor+numSin] = _MI(pdfsource[i],pdfSuspicious[j])

    for i in range(numSin):
        for j in range(i,numSin):
            weightIntens[i+numSor][j+numSor] = val_min #_MI(pdfsink[i],pdfsink[j])
        for j in range(numSus):
            weightIntens[i+numSor][j+numSor+numSin] = _MI(pdfsink[i],pdfSuspicious[j] )


    for i in range(numSus):
        for j in range(i,numSus):
            weightIntens[i+numSor+numSin][j+numSor+numSin] = _MI(pdfSuspicious[i],pdfSuspicious[j])
        
    return weightIntens


def computeSpatialCorrelation(coordSuspicious, numSor, numSin):
    """Computer spacial correlation of ROIs

    Parameters
    ----------
    coordSuspicious: list of tuples
        The Coordination of all testing RIO
    numSor:
        The number of source data point
    numSin:
        The number of sink data point
    
    """

    numSus = len(coordSuspicious)

    weightSpatial = np.zeros((numSus + numSor + numSin,numSus + numSor + numSin),dtype=np.double)

    disSorSor = 0.0
    disSorSin = 1.0
    disSorSus = 0.5
    disSinSin = 0.0
    disSinSus = 0.5   # need to be adjusted

    for i in range(numSor):
        for j in range(numSor):
            weightSpatial[i][j] = disSorSor
        for j in range(numSin):
            weightSpatial[i][j+numSor] = disSorSin
        for j in range(numSus):
            weightSpatial[i][j+numSor+numSin] = disSorSus

    for i in range(numSin):
        for j in range(numSin):
            weightSpatial[i+numSor][j+numSor] = disSinSin
        for j in range(numSus):
            weightSpatial[i+numSor][j+numSor+numSin] = disSinSus

    for i in range(numSus):
        for j in range(numSus):
            weightSpatial[i+numSor+numSin][j+numSor+numSin] = computeSpaDis(coordSuspicious[i],coordSuspicious[j])

    # Normalize susipicious distances 
    val_max = np.max(weightSpatial)
    start = numSor+numSin
    end = numSor+numSin + numSus
    weightSpatial[start:end,start:end] = weightSpatial[start:end,start:end]/val_max
       
    return weightSpatial

def computeSpaDis(coord_1,coord_2):
    """Computer spacial correlation of two ROIs

    Parameters
    ----------
    coord_1:
        The coordination of the first ROI
    coord_2:
        The coordination of the second ROI
    
    """
	
    spatio = 5
    spaDis = np.sqrt((coord_1[0] - coord_2[0])**2 + (coord_1[1] - coord_2[1])**2 + (spatio*coord_1[2] - spatio*coord_2[2])**2 )

    return spaDis

    
def computeWeight(sliceList, sourceSams, sinkSams, alpha, beta = 0.1):
    """Computer the weigh between ROIs using intensity, coordination and features.

    Parameters
    ----------
    sliceList:
        List of suspicious data
    sourceSams:
        List positive data
    sinkSams:
        List of negative data
    alpha:
        Weight coefficient for feature correlation
    beta:
        Weight coefficient for spatial correlation
    
    """


    numSource = len(sourceSams)
    numSink = len(sinkSams)
    numSuspicious = 0
    for i in range(len(sliceList)):
        numSuspicious = numSuspicious + len(sliceList[i].LightPatchList)

    numNode = numSource + numSink + numSuspicious

    ## prepare data for the following process
    numFeat = sliceList[0].feats.shape[1]
    # source data
    featSource = np.zeros((numSource,numFeat),np.double)
    intenSource = []
    cenSource = []
    for i in range(numSource):
        featSource[i] = sourceSams[i].feats
        intenSource.append(sourceSams[i].pdata)
        cenSource.append(sourceSams[i].patch_center)
        
    # sink data
    featSink = np.zeros((numSink,numFeat),np.double)
    intenSink = []
    cenSink = []
    for i in range(numSink):
        featSink[i] = sinkSams[i].feats
        intenSink.append(sinkSams[i].pdata)
        cenSink.append(sinkSams[i].patch_center)

    # suspicious data
    featSuspicious = np.zeros((numSuspicious,numFeat),np.double)
    intenSuspicious = []
    coordSuspicious = []
    cenSuspicious = []
    counter = 0
    for i in range(len(sliceList)):
        
        numPatch = len(sliceList[i].LightPatchList)
        featSuspicious[counter:counter + numPatch] = sliceList[i].feats
        counter = counter + numPatch

        for j in range(numPatch):
            intenSuspicious.append(sliceList[i].LightPatchList[j].pdata)
            coordSuspicious.append(sliceList[i].LightPatchList[j].image_center)
            cenSuspicious.append(sliceList[i].LightPatchList[j].patch_center)
            
    #outputPath = 'C:/Tomosynthesis/localtest/res/'
    #np.savetxt(outputPath + 'featSuspicious.txt', featSuspicious, delimiter='\t')
    #np.savetxt(outputPath + 'featSource.txt', featSource, delimiter='\t')
    #np.savetxt(outputPath + 'featSink.txt', featSink, delimiter='\t')
    
    # compute feature correlation    
    weightFeat = computeFeatureCorrelation(featSuspicious, featSource, featSink)
    #np.savetxt(outputPath + 'featcorrelation.txt', weightFeat, delimiter='\t')
    print 'done feature correlation'

    # compute spatial connection
    weightSpatial = computeSpatialCorrelation(coordSuspicious, numSource, numSink)
    #np.savetxt(outputPath + 'spatcorrelation.txt', weightSpatial, delimiter='\t')
    print 'done spatial correlation'
    
       
    # compute intensity mutual information  
    weightInten = computeIntensityCorrelation(intenSuspicious, intenSource, intenSink,cenSuspicious, cenSource, cenSink)
    #np.savetxt(outputPath + 'mutualinformation.txt', weightInten, delimiter='\t')
    print 'done intensity correlation'

    #regularized weight
    weight = alpha*weightFeat + beta*weightSpatial #+ weightInten
    weight = np.max(weight) - weight
    #np.savetxt(outputPath + 'weight.txt', weight, delimiter='\t')

    return weight


def buildGraph(numSource,numSink,numSuspicious,weight):
    """Build a graph with all positive data points as source, and all negative
    data points as sink.

    Parameters
    ----------
    
    numSource: integer
        The number of positive data points
    numSink: integer
        The number of negative data points
    numSuspicious: integer
        The number of suspicious data points
    weight: numpy array (2D)
        The weight matrix
    """

    numTot = numSource + numSink + numSuspicious
    
    # symmytrize weight
    for i in range(numTot):
        for j in range(0,i):
            weight[i,j] = weight[j,i]

    #outputPath = 'C:/Tomosynthesis/localtest/res/'
    #np.savetxt(outputPath + 'weight_symmetrical.txt', weight, delimiter='\t')
    
    # build graph
    gr = digraph()
    for i in range(numTot):
        gr.add_nodes([i])

    
    for i in range(1,numTot):
        if i == numSource:
            continue
        else:
            # connect source to other nodes except sink
            gr.add_edge((0,i), weight[0][i])
            
            # connect nodes to the sink 
            gr.add_edge((i,numSource), weight[numSource][i])
            
            # connect all other nodes to each other, make sure no loop
            for j in range(i+1,numTot):
                if j == numSource:
                    continue
                else:
                    gr.add_edge((i,j), weight[j][i])

    return gr

def classify(gr,numSource,numSink):
    """Classify using the built graph by the max_flow algorithm

    Parameters
    ----------
    
    gr: 
        The graph model
    numSource: integer
        The number of positive data points
    numSink: integer
        The number of negative data points
    """
    
    flows, cuts = maximum_flow(gr, 0, numSource)

    return cuts

    
def mainClassify(sliceList,sourceSams,sinkSams,alpha = 1,beta = 0.2):
    """Main function for classification using graph cut.

    Parameters
    ----------
    sliceList:
        List of suspicious data
    sourceSams:
        List positive data
    sinkSams:
        List of negative data
    alpha:
        Weight coefficient for feature correlation
    beta:
        Weight coefficient for spatial correlation

    """ 
    
    numSource = len(sourceSams)
    numSink = len(sinkSams)
    numSuspicious = 0
    for i in range(len(sliceList)):
        numSuspicious = numSuspicious + len(sliceList[i].LightPatchList)

    numNode = numSource + numSink + numSuspicious
    
    #compute weight function
    print 'Computing weight ...'
    weight = computeWeight(sliceList, sourceSams, sinkSams, alpha, beta)
    
    
    #build graph
    print 'building graph...'
    gr = buildGraph(numSource,numSink,numSuspicious,weight)
    
    #classify
    print 'Classifying...'
    predicts = classify(gr,numSource,numSink)

    return predicts


    
    
