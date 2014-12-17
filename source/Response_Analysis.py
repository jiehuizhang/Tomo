"""This file includes functions that analyzing the filtering responses.
Specifically, creating batch response, taking the voting procedure,etc.
"""

import numpy as np
import sklearn.preprocessing

def dump_vector(response):
    """
    This function reorgazies the responses derived from the same kernel set
    into an numpy array. (For debugging use only)
    """
    
    shape = response[0].shape
    response_var = np.zeros(shape, dtype=np.double)
    for row in range(shape[0]):
        for col in range(shape[1]):
            temp_vec = np.zeros(len(response), dtype=np.double)
            for k in range(len(response)):
                temp_vec[k] = response[k][row][col]
            response_var[row][col] = np.var(temp_vec)

    return response_var

def cerat_batch_response(response,sampRate,winSize):
    """This function creats the batch response matrix in which each element
    is the mean value of a neighbour area from the corresponding response matrix

    Parameters
    ----------
    response: numpy array
        Input response matrix
    sampRate: integer
        Sampling rate
    winSize: integer
        Size of the neighbourhood   
    """
    
    numResp = len(response)
    shape = response[0].shape
    nrow = shape[0]
    ncol = shape[1]

    batchResp = []
    integratedResp = []
    for k in range(numResp):
        rows = np.array(range(sampRate,nrow,sampRate))
        cols = np.array(range(sampRate,ncol,sampRate))
        rsu = np.maximum(rows - winSize,np.zeros(len(rows), dtype=np.int))
        rsd = np.minimum(rows + winSize,np.ones(len(rows), dtype=np.int)*(nrow-1))
        csl = np.maximum(cols - winSize,np.zeros(len(cols), dtype=np.int))
        csr = np.minimum(cols + winSize,np.ones(len(cols), dtype=np.int)*(ncol-1))
        temp_resp = np.zeros((len(rows),len(cols)), dtype=np.double)
        temp_intresp = np.zeros(shape, dtype=np.double)
        for rs in range(len(rows)):
            for cs in range(len(cols)):
                mean_var = np.sum(response[k][rsu[rs]:rsd[rs],csl[cs]:csr[cs]])
                temp_resp[rs][cs] =  mean_var
                temp_intresp[rsu[rs]:rsd[rs],csl[cs]:csr[cs]] = mean_var
        batchResp.append(temp_resp)
        integratedResp.append(temp_intresp)

    return (batchResp, integratedResp)

def integrating_poll(response,sampRate,winSize,shape):
    """
    This function reverse the sampling proceduring after the
    voting result is derived

    Parameters
    ----------
    
    response: numpy_array
        Input voting result
    sampRate: integer
        sampling rate
    winSize: integer
        neighbourhood size
    """

    nrow = shape[0]
    ncol = shape[1]
        
    rows = np.array(range(sampRate,nrow,sampRate))
    cols = np.array(range(sampRate,ncol,sampRate))
    rsu = np.maximum(rows - winSize,np.zeros(len(rows), dtype=np.int))
    rsd = np.minimum(rows + winSize,np.ones(len(rows), dtype=np.int)*(nrow-1))
    csl = np.maximum(cols - winSize,np.zeros(len(cols), dtype=np.int))
    csr = np.minimum(cols + winSize,np.ones(len(cols), dtype=np.int)*(ncol-1))

    inte_poll = np.zeros(shape, dtype=np.double)
    for rs in range(len(rows)):
        for cs in range(len(cols)):
            inte_poll[rsu[rs]:rsd[rs],csl[cs]:csr[cs]] = response[rs][cs]
                
    return inte_poll


def getCDF(temp_response):
    """
    This function calculates voting score of response magnitude.
    The score is calculated as the rank of sorted magnitude.

    Parameters
    ----------
    temp_response: numpy array
        Input batch response matrix
    """
    
    shape = temp_response.shape
    nrow = shape[0]
    ncol = shape[1]
    len_hist = nrow*ncol
    resp_arr = temp_response.reshape((len_hist,))
    sorted_arr = np.sort(resp_arr)
    
    cdf = np.zeros(shape, dtype=np.double)
    for r in range(nrow):
        for c in range(ncol):
            index = np.where(sorted_arr==temp_response[r,c])
            cdf[r,c] = np.double(index[0][0])/np.double(len_hist)
   
    return cdf

def getODF(response, threshold = 0.1):
    """
    This function calculate the orientation voting score.
    The score in each orientation is set to 1 if it is above the
    percentage threshold, other wise set to zero

    Parameters
    ----------  
    response: numpy array (2D)
        Input batch response
    threshold = 0.2
        for num_orientation = 4
    threshold = 0.1
        for num_orientation = 8
    """
    shape = response[0].shape
    numResp = len(response)
    epsilon=0.00001
    
    odf = []
    for k in range(len(response)):
        odf.append(np.zeros(shape, dtype=np.double))

    for row in range(shape[0]):
        for col in range(shape[1]):
            temp_vec = np.zeros(numResp, dtype=np.double)
            for k in range(numResp):
                temp_vec[k] = response[k][row][col]
            if abs(sum(temp_vec)) < epsilon:
                continue
            for k in range(numResp):
                var = np.double(response[k][row][col])/sum(temp_vec)
                if var > threshold:
                    odf[k][row][col] = 1.0
    return odf
    
def vote(response,alpha = 1.1):
    """
    This function procedes the voting procedure.
    Magnitude score are calculated in function getCDF
    Orientation score are calculated in function getODF

    Parameters
    ----------   
    response:
        Input of batch response
    alpha:
        weight of orientation score
        alpha = 1.1 for num_orientation = 4
        alpha = 0.55 for num_orientation = 8
    """
    
    numResp = len(response)
    shape = response[0].shape
    nrow = shape[0]
    ncol = shape[1]

    cdf= []
    for k in range(numResp):
        cdf.append(getCDF(response[k]))

    odf = getODF(response)
        
    poll_intensity = np.zeros(shape, dtype=np.double)
    poll_orientation = np.zeros(shape, dtype=np.double)

    for k in range(numResp):
        # calculate intensity score
        poll_intensity = poll_intensity + cdf[k]

        # calculate orientation score
        poll_orientation = poll_orientation + odf[k]

    return alpha*poll_orientation + poll_intensity








    
