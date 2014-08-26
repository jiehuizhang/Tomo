""" Build Graphs based on ROI """
import numpy as np

import ImageIO
import TImage
import TPatch

from pygraph.classes.graph import graph
from pygraph.classes.digraph import digraph
from pygraph.algorithms.minmax import minimal_spanning_tree,\
shortest_path, heuristic_search, shortest_path_bellman_ford, maximum_flow, cut_tree
    
def computeFeatureCorrelation(featSuspicious, featSource, featSink):

    numSus = featSuspicious.shape[0]
    numSor = featSource.shape[0]
    numSin = featSink.shape[0]

    weightFeat = np.zeros((numSus + numSor + numSin,numSus + numSor + numSin),dtype=np.double)

    for i in range(numSor):
        for j in range(numSor):
            weightFeat[i][j] = computeDis(featSource[i],featSource[j])
        for j in range(numSin):
            weightFeat[i][j+numSor] = computeDis(featSource[i],featSink[j])
        for j in range(numSus):
            weightFeat[i][j+numSor+numSin] = computeDis(featSource[i],featSuspicious[j])

    for i in range(numSin):
        for j in range(numSor):
            weightFeat[i+numSor][j] = computeDis(featSink[i],featSource[j])
        for j in range(numSin):
            weightFeat[i+numSor][j+numSor] = computeDis(featSink[i],featSink[j])
        for j in range(numSus):
            weightFeat[i+numSor][j+numSor+numSin] = computeDis(featSink[i],featSuspicious[j])

    for i in range(numSus):
        for j in range(numSor):
            weightFeat[i+numSor+numSin][j] = computeDis(featSuspicious[i],featSource[j])
        for j in range(numSin):
            weightFeat[i+numSor+numSin][j+numSor] = computeDis(featSuspicious[i],featSink[j])
        for j in range(numSus):
            weightFeat[i+numSor+numSin][j+numSor+numSin] = computeDis(featSuspicious[i],featSuspicious[j])
        
    return weightFeat        
    
def computeDis(feat1,feat2,opt = 'Euc'):

    dis = 0.0
    if opt == 'Euc':
        """Euclidean distance"""    
        for i in range(feat1.shape[0]):
            dis = dis + (feat1[i] - feat2[i])**2
        dis = np.sqrt(dis)
        
    return dis

def computeMutual(data1, data2, w = 3):

    m,n = data1.shape
    numTot = (2*w + 1)**2
    
    for p in range(w,m - w):
        for q in range(w,n-w):
            
            # extract window
            sub1 = data1[p-w:p+w+1,q-w:q+w+1]
            sub2 = data2[p-w:p+w+1,q-w:q+w+1]

            #compute mutual information
            if np.sum(sub1 - sub2) == 0:
                muinfo = 1
            else:
                
                #normalize sub1
                max1 = np.max(sub1)
                min1 = np.min(sub1)
                print sub1.shape
                if max1 == min1:
                    sub1 = np.ones(numTot,np.double)
                else:
                    sub1 = sub1.ravel()
                    sub1 = np.double((sub1 - min1))/np.double((max1 - min1))

                #normalize sub2
                max2 = np.max(sub2)
                min2 = np.min(sub2)
                if max2 == min2:
                    sub2 = np.ones(numTot,np.double)
                else:
                    sub2 = sub2.ravel()
                    sub2.reshape((numTot,))
                    sub2 = np.double((sub2 - min2))/np.double((max2 - min2))

                print sub1.shape
                #get pdf
                pdf1 = np.double(sub1)/np.double(np.sum(sub1))
                pdf2 = np.double(sub2)/np.double(np.sum(sub2))

                # get cdf
                cdf1 = np.ones(numTot,np.double)
                cdf2 = np.ones(numTot,np.double)
                cdf1[0] = pdf1[0]
                cdf2[0] = pdf2[0]
                for i in range(1,numTot):
                    cdf1[i] = pdf1[i] + cdf1[i-1]
                    cdf2[i] = pdf2[i] + cdf2[i-1]
                    
                #pearson correlation between marginal pdfs
                temp1 = pdf1 - np.mean(pdf1)
                temp2 = pdf2 - np.mean(pdf2)
                if np.sum(temp1*temp2)==0:
                    rho = 0
                else:
                    rho = np.sum(temp1*temp2)/np.sqrt(np.sum(temp1*temp1)*np.sum(temp2*temp2))

                #population standard deviation
                e_pdf1 = 0
                e2_pdf1 = 0
                e_pdf2 = 0
                e2_pdf2 = 0
                for i in range(numTot):
                    e_pdf1  = e_pdf1  + i*e_pdf1[i];
                    e2_pdf1  = e2_pdf1  + i**2*e2_pdf1[i];
                    e_pdf2  = e_pdf2 + i*e_pdf2[i];
                    e2_pdf2  = e2_pdf2  + i**2*e2_pdf2[i];

                sd1 = np.sqrt(e2_pdf1 - e_pdf1**2)
                sd2 = np.sqrt(e2_pdf2 - e_pdf2**2)

                #joint entropy
                if rho >= 0:
                    if rho == 0 or sd1 == 0 or sd2 == 0:
                        phi = 0
                    else:
                        covup = 0
                        for i in range(numTot):
                            for i in range(numTot):
                                covup = covup + 0.5(cdf2[i] + cdf1[j] - np.abs(cdf2[i]-cdf1[j])) - cdf2[i]*cdf1[j]

                        corrUp = covup/(sd1*sd2)
                        phi = rho/corrUp

                    jpdfUp = 0.5*(cdf2[0]+cdf1[0]-np.abs(cdf2[0]-cdf1[0]))
                    jpdf = phi*jpdfUp + (1-phi)*pdf2[0]*pdf1[0]
                    if jpdf != 0:
                        jointEntropy = real(-jpdfnp.log2(jpdf))

                    # 1-d boundaries
                    for i in range(1,numTot):
                        jpdfUp = 0.5*(cdf2[i]+cdf1[0]-np.abs(cdf2[i]-cdf1[0])) - 0.5*(cdf2[i-1]+cdf1[0]-np.abs(cdf2[i-1]-cdf1[0]))
                        jpdf = phi*jpdfUp + (1-phi)*pdf2[i]*pdf[0]
                        if jpdf != 0:
                            jointEntropy = jointEntropy + real(-jpdf*np.log2(jpdf))

                    for j in range(1,numTot):
                        jpdfUp = 0.5*(cdf2[0]+cdf1[j]-np.abs(cdf2[0]-cdf1[j])) - 0.5*(cdf2[0]+cdf1[j-1]-np.abs(cdf2[0]-cdf[j-1]))
                        jpdf = phi*jpdfUp + (1-phi)*pdf2[0]*pdf1[j];
                        if jpdf != 0:
                            jointEntropy = jointEntropy + real(-jpdf*np.log2(jpdf))

                        # 2-D walls
                    for i in range(1,numTot):
                        for j in range(1,numTot):
                            jpdfUp = 0.5*(cdf2[i]+cdf1[j]-np.abs(cdf2[i]-cdf1[j])) - 0.5*(cdf2[i-1]+cdf1[j]-np.abs(cdf2[i-1]-cdf1[j])) - 0.5*(cdf2[i]+cdf1[j-1]-np.abs(cdf2[i]-cdf1[j-1])) + 0.5*(cdf2[i-1]+cdf1[j-1]-np.abs(cdf2[i-1]-cdf1[j-1]))
                            jpdf = phi*jpdfUp + (1-phi)*pdf2[i]*pdf1[j]
                            if jpdf != 0:
                                jointEntropy = jointEntropy + real(-jpdf*np.log2(jpdf))

                if rho < 0:

                    if sd1 == 0 or sd2 == 0:
                        theta = 0
                    else:
                        covLo = 0
                        for i in range(1,numTot):
                            for j in range(1,numTot):
                                covLo = covLo + 0.5*(cdf2[i]+cdf1[j]-1+np.abs(cdf2[i]+cdf1[j]-1)) - cdf2[i]*cdf1[j]

                        corrLo = covLo/(sd1*sd2)
                        theta = rho/corrLo

                    jpdfLo = 0.5*(cdf2[0]+cdf1[0]-1+np.abs(cdf2[0]+cdf1[0]-1))
                    jpdf = theta*jpdfLo + (1-theta)*pdf2[0]*pdf1[0]
                    if jpdf != 0:
                        jointEntropy = real(-jpdf*np.log2(jpdf))

                    # 1-d boundaries
                    for i in range(1,numTot):
                        jpdfLo = 0.5*(cdf2[i]+cdf1[0]-1+np.abs(cdf2[i]+cdf1[0]-1)) - 0.5*(cdf2[i-1]+cdf1[0]-1+np.abs(cdf2[i-1]+cdf1[0]-1))
                        jpdf = theta*jpdfLo + (1-theta)*pdf2[i]*pdf1[0]
                        if jpdf != 0:
                            jointEntropy = jointEntropy + real(-jpdf*np.log2(jpdf))

                    for j in range(1,numTot):
                        jpdfLo = 0.5*(cdf2[0]+cdf1[j]-1+np.abs(cdf2[0]+cdf1[j]-1)) - 0.5*(cdf2[0]+cdf1[j-1]-1+np.abs(cdf2[0]+cdf1[j-1]-1))
                        jpdf = theta*jpdfLo + (1-theta)*pdf2[0]*pdf1[j]
                        if jpdf != 0:
                            jointEntropy = jointEntropy + real(-jpdf*np.log2(jpdf))

                    # 2-D walls
                    for i in range(1,numTot):
                        for j in range(1,numTot):
                            jpdfLo = 0.5*(cdf2[i]+cdf1[j]-1+np.abs(cdf2[i]+cdf1[j]-1)) - 0.5*(cdf2[i-1]+cdf1[j]-1+np.abs(cdf2[i-1]+cdf1[j]-1)) - 0.5*(cdf2[i]+cdf1[j-1]-1+np.abs(cdf2[i]+cdf1[j-1]-1)) + 0.5*(cdf2[i-1]+cdf1[j-1]-1+np.abs(cdf2[i-1]+cdf1[j-1]-1))
                            jpdf = theta*jpdfLo + (1-theta)*pdf2[i]*pdf1[j]
                            if jpdf != 0:
                                jointEntropy = jointEntropy + real(-jpdf*np.log2(jpdf))

                # Marginal entropies
                index = pdf1 != 0
                Entropy1 = np.sum(-pdf1(index)*np.log2(pdf1(index)))
                index = pdf2 != 0
                Entropy2 = np.sum(-pdf2(index)*np.log2(pdf2(index)))
            
                # Mutual information 
                muinfo = Entropy1 + Entropy2 - jointEntropy;  
            
                # Overall normalized mutual information
                if muinfo == 0:
                    muinfo = 0
                else:
                    muinfo = 2*muinfo/(Entropy1+Entropy2)


    return muinfo

                
def computeIntensityCorrelation(intenSuspicious, intenSource, intenSink):

    numSus = len(intenSuspicious)
    numSor = len(intenSource)
    numSin = len(intenSink)

    weightIntens = np.zeros((numSus + numSor + numSin,numSus + numSor + numSin),dtype=np.double)

    for i in range(numSor):
        for j in range(numSor):
            weightIntens[i][j] = computeMutual(intenSource[i],intenSource[j])
        for j in range(numSin):
            weightIntens[i][j+numSor] = computeMutual(intenSource[i],intenSink[j])
        for j in range(numSus):
            weightIntens[i][j+numSor+numSin] = computeMutual(intenSource[i],intenSuspicious[j])

    for i in range(numSin):
        for j in range(numSor):
            weightIntens[i+numSor][j] = computeMutual(intenSink[i],intenSource[j])
        for j in range(numSin):
            weightIntens[i+numSor][j+numSor] = computeMutual(intenSink[i],intenSink[j])
        for j in range(numSus):
            weightIntens[i+numSor][j+numSor+numSin] = computeMutual(intenSink[i],intenSuspicious[j])

    for i in range(numSus):
        for j in range(numSor):
            weightIntens[i+numSor+numSin][j] = computeMutual(featSuspicious[i],intenSource[j])
        for j in range(numSin):
            weightIntens[i+numSor+numSin][j+numSor] = computeMutual(featSuspicious[i],intenSink[j])
        for j in range(numSus):
            weightIntens[i+numSor+numSin][j+numSor+numSin] = computeMutual(featSuspicious[i],featSuspicious[j])
        
    return weightIntens


def computeSpatialCorrelation(coordSuspicious, numSor, numSin):

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
        for j in range(numSor):
            weightSpatial[i+numSor][j] = disSorSin
        for j in range(numSin):
            weightSpatial[i+numSor][j+numSor] = disSinSin
        for j in range(numSus):
            weightSpatial[i+numSor][j+numSor+numSin] = disSinSus

    for i in range(numSus):
        for j in range(numSor):
            weightSpatial[i+numSor+numSin][j] = disSorSus
        for j in range(numSin):
            weightSpatial[i+numSor+numSin][j+numSor] = disSinSus
        for j in range(numSus):
            weightSpatial[i+numSor+numSin][j+numSor+numSin] = computeSpaDis(coordSuspicious[i],coordSuspicious[j])
        
    return weightSpatial

def computeSpaDis(coord_1,coord_2):

    spaDis = np.sqrt((coord_1[0] - coord_2[0])**2 + (coord_1[1] - coord_2[1])**2 + (10*coord_1[2] - 10*coord_2[2])**2 )

    return spaDis

    
def computeWeight(sliceList, sourceSams, sinkSams, alpha, beta):

    numSource = len(sourceSams)
    numSink = len(sinkSams)
    numSuspicious = 0
    for i in range(len(sliceList)):
        numSuspicious = numSuspicious + len(sliceList[i].LightPatchList)

    numNode = numSource + numSink + numSuspicious

    ## prepare data for following process
    numFeat = sliceList[0].feats.shape[1]
    # source data
    featSource = np.zeros((numSource,numFeat),np.double)
    intenSource = []    
    for i in range(numSource):
        featSource[i] = sourceSams[i].feats
        intenSource.append(sourceSams[i].pdata)
        
    # sink data
    featSink = np.zeros((numSink,numFeat),np.double)
    intenSink = []
    for i in range(numSink):
        featSink[i] = sinkSams[i].feats
        intenSink.append(sinkSams[i].pdata)    

    # suspicious data
    featSuspicious = np.zeros((numSuspicious,numFeat),np.double)
    intenSuspicious = []
    coordSuspicious = []   
    counter = 0
    for i in range(len(sliceList)):
        
        numPatch = len(sliceList[i].LightPatchList)
        featSuspicious[counter:counter + numPatch] = sliceList[i].feats
        counter = counter + numPatch

        for j in range(numPatch):
            intenSuspicious.append(sliceList[i].LightPatchList[j].pdata)
            coordSuspicious.append(sliceList[i].LightPatchList[j].image_center)
    
    # compute feature correlation    
    weightFeat = computeFeatureCorrelation(featSuspicious, featSource, featSink)
       
    # compute intensity mutual information  
    weightInten = computeIntensityCorrelation(intenSuspicious, intenSource, intenSink)
      
    # compute spatial connection
    weightSpatial = computeSpatialCorrelation(coordSuspicious, numSource, numSink)

    #regularized weight
    weight = weightInten + alpha*weightFeat + beta*weightSpatial

    return weight


def buildGraph(numSource,numSink,numSuspicious,weight):

    numTot = numSource + numSink + numSuspicious
    gr = digraph()
    for i in range(numTot):
        gr.add_nodes(i)

    
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

def classify(gr,numSource):
    
    flows, cuts = maximum_flow(gr, 0, numSource)

    pred = cuts[numSource + numSource:-1]

    return pred

    
def mainClassify(sliceList,sourceSams,sinkSams,alpha = 1,beta = 1):
    
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


    
    
