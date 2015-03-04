'''
Creating Graph Model for combining features


@author: myousefi
'''
import scipy
import numpy as np
#import pyqtgraph
import networkx as nx
import entropy_estimators as ee
import scipy.spatial as ss

from scipy import spatial
import RBST
import waveletcLBP
import cv2
import ImageIO
import systems
import utils


'''x, y = np.mgrid[0:4, 0:4]
points = zip(x.ravel(), y.ravel()) 
tree = spatial.cKDTree(points)
 
tree.query_ball_point([2, 0], 1)
 
 

 
G=nx.Graph()
G.add_node(1)
G.add_edge(1,2)
x = [[1.3],[3.7],[5.1],[2.4],[3.4]]
y = [[1.5],[3.32],[5.3],[2.3],[3.3]]
weight=ee.mi(x,y)
G.add_edge(1,2, weight)
print weight'''


#def graphimage():
def weightinginfo(feature1,feature2,im1,im2):
    #weight=ee.mi(feature1,feature2)
    '''X=feature1
    Y=feature2
    Xn=0
    Xm=10
    Yn=0
    Ym=10
    X=np.asarray(X)
    Y=np.asarray(Y)
    X=utils.quantise(X, 10, uniform='sampling', minmax=None, centers=False)
    print X
    Y=utils.quantise(Y, 10, uniform='sampling', minmax=None, centers=False)
    sys = systems.DiscreteSystem(X,(Xn,Xm),Y,(Yn,Ym))
    sys.calculate_entropies(method='qe',calc=['HX','HXY','HiXY','HshXY'])
    weight=sys.I()'''
    
    feature1=feature1.tolist()
    
    feature2=feature2.tolist()
    weight=ee.kldiv(feature1,feature2,k=3,base=2)
    print 'weight is', weight
    return weight

'''finding longest path in graph for optimization mutual information'''
def longest_path(G):
    dist = {} # stores [node, distance] pair
    for node in nx.topological_sort(G):
        # pairs of dist,node for all incoming edges
        pairs = [(dist[v][0]+1,v) for v in G.pred[node]] 
        if pairs:
            dist[node] = max(pairs)
        else:
            dist[node] = (0, node)
    node,(length,_)  = max(dist.items(), key=lambda x:x[1])
    path = []
    while length > 0:
        path.append(node)
        length,node = dist[node]
    return list(reversed(path))


def graphrepresentation(myarray,im):
    '''creating a graph and add images in tiff image as nodes of graphs  '''
    G=nx.Graph()
    Size=myarray.shape
    a=Size[0]
    for i in range(0,a-1):
        G.add_node(i,point=myarray[i])
        
    #G.add_nodes_from(im)
    
    #G.add_nodes_from(myarray)
    '''after create graph we should add edges and weight to edges by mutual information  '''
    numberofedges=myarray.shape[0]
    
    
    for i in range (myarray.shape[0]):
        sum=0  
        path='/Users/Shared/TomosynthesisData/processed/5039/'
        fileName='5039.tif'
        fileName = fileName.split('.')
        fileName = fileName[0] + str(i) + '.tif'
        im1 = ImageIO.imReader(path,fileName,'tif',2)
        for j in range(myarray.shape[0]):
            G.add_edge(i,j)
            #im1=im[i]
            fileName='5039.tif'
            fileName = fileName.split('.')
            fileName = fileName[0] + str(j) + '.tif'
            im2 = ImageIO.imReader(path,fileName,'tif',2)
            
            '''finding mutual information of different features then add them for finding weight'''
            feature1=waveletcLBP.concatinationhist(im1.data[0])
            feature2=waveletcLBP.concatinationhist(im2.data[0])
            weight=weightinginfo(feature1,feature2,im1,im2)
            sum=sum+weight
            
            feature1=RBST.shortrunfacility(im1.data[0],im1.data[0])
            feature2=RBST.shortrunfacility(im2.data[0],im2.data[0]) 
            weight=weightinginfo(feature1,feature2,im1,im2)
            
            
            print len(feature1), len(feature2)
            
            
            feature1=RBST.shortrunfacility(im1.data[0],im1.data[0])
            feature2=RBST.shortrunfacility(im2.data[0],im2.data[0]) 
            weight=weightinginfo(feature1,feature2,im1,im2)
            sum=sum+weight
            
            G.add_weighted_edges_from([i,j,sum])
            
    listpath=longest_path(G)
    "after finding longest we can combine features with the weighting list and order in list"
    lengthlist=len(listpath)       
    for i in range(lengthlist-1):
        a=listpath(i)
        b=listpath(i+1)
        weightonthelist=G.adjacency_ite(a,b)
        weightlist=list.append(weightonthelist)
    return weightlist, listpath    
        
            
             
        
        
    
    
     
    
    
    
    