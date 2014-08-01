"""Micro cacification detection"""
import math
import numpy as np
from scipy import ndimage
import scipy.ndimage.filters as filters
import tiffLib
import TMicroCal

def bs_Estimatimation(imdata, threshold, depth = 16):

    height, width = imdata.shape
    scale = int(math.pow(2,depth))
    npixel = width*height
    
    # Generate histogram
    hist = np.zeros(scale, dtype=np.uint16)
    for j in range(width):
        for i in range(height):
            val = imdata[i][j]
            hist[val] = hist[val] + 1

    # Calculate probability
    prob = np.zeros(scale, dtype=np.float32)
    for i in range(scale):
        prob[i] = float(hist[i])/float(npixel)
    # Generate CDF
    cdf = np.zeros(scale, dtype=np.float32)
    cdf[0] = prob[0]
    for i in range(1,scale-1):
        cdf[i] = prob[i] + cdf[i-1]
        if cdf[i]> threshold:
            estimated_bs = i
            break

    return estimated_bs

def fg_thresholding(imdata, threshold):

    fg = np.zeros(imdata.shape, dtype=np.uint16)
    fg[:,:] = imdata[:,:]
    index = fg < threshold
    fg[index] = threshold
    return fg

    
def log_filtering(imdata, winSize,sigma, fg_thresh,option = 'proplog'):
    '''sample_rate should be less than window size'''
    
    sample_rate = winSize
    nrow, ncol = imdata.shape
    rows = np.array(range(winSize,nrow,sample_rate))
    cols = np.array(range(winSize,ncol,sample_rate))

    rsu = np.maximum(rows - winSize,np.zeros(len(rows), dtype=np.int))
    rsd = np.minimum(rows + winSize,np.ones(len(rows), dtype=np.int)*(nrow-1))
    csl = np.maximum(cols - winSize,np.zeros(len(cols), dtype=np.int))
    csr = np.minimum(cols + winSize,np.ones(len(cols), dtype=np.int)*(ncol-1))

    log_response = np.zeros(imdata.shape, dtype=np.double)
    for rs in range(len(rows)):
        for cs in range(len(cols)):
            # extract data
            block = imdata[rsu[rs]:rsd[rs],csl[cs]:csr[cs]]

            # extract foreground
            estimated_bs = bs_Estimatimation(block, fg_thresh)
            fg = fg_thresholding(block,estimated_bs)

            # compute log response 
            temp_response = np.zeros(block.shape, np.double)
            filters.gaussian_laplace(fg, sigma, output=temp_response, mode='reflect')

            # composite response
            
            ub = rsu[rs] + winSize/2
            loc_ub = winSize/2
            db = rsd[rs] - winSize/2 
            loc_db = 3*winSize/2 
            lb = csl[cs] + winSize/2 
            loc_lb = winSize/2 
            rb = csr[cs] - winSize/2 
            loc_rb = 3*winSize/2
            # upper bound
            if rsu[rs] == 0:
                ub = 0
                loc_ub = 0
            # lower bound
            if rsd[rs] == nrow-1:
                db = nrow-1
                loc_db = temp_response.shape[0]
            # left bound
            if csl[cs] == 0:
                lb = 0
                loc_lb = 0
            # right bound
            if csr[cs] == ncol - 1:
                rb = ncol - 1
                loc_rb = temp_response.shape[1]
            
            log_response[ub:db,lb:rb] = temp_response[loc_ub:loc_db, loc_lb:loc_rb]

    if option == 'log':
        return -log_response
    if option == 'proplog':
        log_rep_prop = 1000*log_response / imdata
        return -log_rep_prop

def laebl_connecte_comp(imdata, threshold, size_constrain):

    # calculated connected labels
    mask = imdata > threshold
    label_im, nb_labels = ndimage.label(mask)

    # remove objects out of size_constrain
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mask_size1 = sizes < size_constrain[0]
    mask_size2 = sizes > size_constrain[1]
    mask_size = mask_size1 + mask_size2
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0

    # clear out regions in the original image
    rm_imdata = np.zeros(imdata.shape, dtype=np.double)
    rm_imdata[:,:] = imdata[:,:]
    index = label_im == 0
    rm_imdata[index] = 0

    return rm_imdata

def MC_buildup_2d(im,mc_im):

    mask = mc_im > 0
    label_im, nb_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, label_im, range(1,nb_labels + 1))
    mean_vals = ndimage.sum(im, label_im, range(1, nb_labels + 1))
    mean_vals = mean_vals/sizes
    
    coordinateX_im = np.zeros(im.shape, dtype=np.int)
    for i in range(im.shape[0]):
        coordinateX_im[i,:] = i
    coordinateY_im = np.zeros(im.shape, dtype=np.int)
    for j in range(im.shape[1]):
        coordinateY_im[:,j] = j
    centers_X  = ndimage.sum(coordinateX_im, label_im, range(1, nb_labels + 1))
    centers_Y  = ndimage.sum(coordinateY_im, label_im, range(1, nb_labels + 1))
    centers_X = centers_X/sizes
    centers_Y = centers_Y/sizes

    
    
    mcList = []
    for i in range(nb_labels):
        mc = TMicroCal.TMicroCal()

        mc.area = sizes[i]
        mc.intensity = mean_vals[i]
        mc.center[0] = int(centers_X[i])
        mc.center[1] = int(centers_Y[i])
        lab = label_im[int(centers_X[i])][int(centers_Y[i])]
        mc.label = lab
        slice_x, slice_y = ndimage.find_objects(label_im == lab)[0]
        mc.roi = im[slice_x, slice_y]

        mcList.append(mc)

    return mcList
        

def MC_connect_2d(mcList,dis_threshold):

    for mc in mcList:
        cent = mc.center
        for mc_oth in mcList:
            cent_oth = mc_oth.center

            dis = math.sqrt((cent[0] - cent_oth[0])**2 + (cent[1] - cent_oth[1])**2)
            if dis<dis_threshold:
                mc.neighbours_2d.append(mc_oth.label)
                mc.neighbour_dis_2d.append(dis)
        
        mc.computeDensity_2d()

def MC_connect_3d(mcLists,tolerance = 10):
    
    global_id = 0
    z_size = len(mcLists)
    for i in range(z_size-1):
        for index_curr in range(len(mcLists[i])):
            center = mcLists[i][index_curr].center
            for indec_nxt in range(len(mcLists[i+1])):
                neighb_cen = mcLists[i+1][indec_nxt].center
                dis = math.sqrt((center[0] - neighb_cen[0])**2 + (center[1] - neighb_cen[1])**2)
                if dis < tolerance:
                    if mcLists[i][index_curr].global_flag == False:
                        mcLists[i][index_curr].global_id = global_id
                        global_id = global_id + 1
                    mcLists[i+1][indec_nxt].global_id = mcLists[i][index_curr].global_id
                    mcLists[i+1][indec_nxt].global_flag = True              
    return global_id

def MCs_constuct_3d(mcLists,global_id):

    '''initilize lists '''
    global_list = []
    for i in range(global_id):
        templist = []
        global_list.append(templist)
        
    for i in range(len(mcLists)):
        for item in mcLists[i]:
            index = item.global_id
            if index != None:
                global_list[index].append(item)

    return global_list

def MCs_constrain(global_list, num_neighbour = 3):

    MC_List_3D = []
    for item in global_list:
        if len(item) >= num_neighbour:
            mc3d = TMicroCal.TMicroCal_3D()
            cenx, ceny, cenz = 0,0,0
            intensity = 0
            volume = 0
            for mc2d in item:
                cenx = cenx + mc2d.center[0]
                ceny = ceny + mc2d.center[1]
                cenz = cenz + mc2d.center[2]
                intensity = intensity + mc2d.intensity*mc2d.area
                volume = volume + mc2d.area

            length = len(item)
            cenx = cenx/length
            ceny = ceny/length
            cenz = cenz/length
            
            mc3d.center = (int(cenx),int(ceny),int(cenz))
            mc3d.volume = volume
            mc3d.intensity = intensity/volume

            MC_List_3D.append(mc3d)

    return MC_List_3D

def parallel_MC_Detection(i,imdata):
    
    log = log_filtering(imdata,winSize=40,sigma=3,fg_thresh = 0.6)
    constrained_log = laebl_connecte_comp(log,threshold=3.0,size_constrain = (2,80))
    mcList = MC_buildup_2d(imdata,constrained_log)
    MC_connect_2d(mcList,dis_threshold = 300)
    for mc_item in mcList:
        mc_item.center[2] = i

    return mcList
                       
def parallelWrapper(args):
    
    return parallel_MC_Detection(*args)
        
    
                    
                
            
            
        
        
    

    
    





            
