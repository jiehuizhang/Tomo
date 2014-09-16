import numpy as np
#import mlpy.wavelet as wave
import pywt
import mahotas 
import matplotlib.pyplot as plt
 
def pathestoforth(imdata):
    Size=imdata.shape
    width=Size[1]
    heigth=Size[0]
    Arpart=np.split(imdata,4)
    #print Arpart
    p1=Arpart[0]
    p2=Arpart[1]
    p3=Arpart[2]
    p4=Arpart[3]
    #print p1
    return p1,p2,p3,p4

def LocalbinaryPattern(imdata):
    p1,p2,p3,p4=pathestoforth(imdata)
    His1=mahotas.features.lbp(p1,1,8,False)
    His2=mahotas.features.lbp(p2,1,8,False)
    His3=mahotas.features.lbp(p3,1,8,False)
    His4=mahotas.features.lbp(p4,1,8,False)
    #PartH=[His1,His2,His3,His4]
    #matrix=np.ndarray((4,), dtype=float, order='F')
    #print matrix
    #np.r_[matrix, His1]
    
    print len(His2)
    matrix=np.zeros(shape=(4,36))
    #matrix=np.vstack([matrix,His1])
    matrix[0]=His1
    matrix[1]=His2
    matrix[2]=His3
    matrix[3]=His4 
    
    print matrix.shape
    PartH=matrix
    ## mahotas.features.lbp.lbp_transform(p1,1,8,False,true)
    return PartH


def waveletLocalbinarypattern(imdata):
    x=imdata
    ##Subimdata=wave.dwt(x=x, wf='d', k=4)
    subimdata=pywt.wavedec2(x, 'db1', level=2)
    cA2, (cH2, cV2, cD2), (cH1, cV1, cD1)=subimdata
    plt.plot(cH2)
    w1=cH1+cV1+cD1
    plt.plot(w1)
    w2=cH2+cV2+cD2
    return cA2,w1,w2


def concatinationhist(imdata):
     W1,W2,Wll=waveletLocalbinarypattern(imdata)
     PartH1=LocalbinaryPattern(W1)
     PartH2=LocalbinaryPattern(W2)
     PartHll=LocalbinaryPattern(Wll)
     #Finfeature=[PartH1,PartH2,PartHll]
     matrix=np.zeros(shape=(12,36))
     matrix[0]=PartH1[0]
     matrix[1]=PartH1[1]
     matrix[2]=PartH1[2]
     matrix[3]=PartH1[3]
     
     matrix[4]=PartH2[0]
     matrix[5]=PartH2[1]
     matrix[6]=PartH2[2]
     matrix[7]=PartH2[3]
     
     matrix[8]=PartHll[0]
     matrix[9]=PartHll[1]
     matrix[10]=PartHll[2]
     matrix[11]=PartHll[3]   
     Finfeature=matrix
     
     
     
     return Finfeature

    
    



    
     
