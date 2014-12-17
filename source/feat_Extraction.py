import TPatch

def patch_Extraction(im, poll, zslice, sampRate, patch_size, threshold = 11.7):
    """
    This function extracts region of interest from the original image 
    based on the voting analysis result.

    Parameters
    ----------
    im:
        original gray level image
    poll:
        voting result
    zslice:
        the slice index of the current image slice in the stack
    sampRate:
        sampling rate
    patch_size:
        User defined ROI size
    threshold:
        threshold of voting score 
                threshold = 7.5(not nesseary optimal) if num_orientation = 4
                threshold = 16.4 if num_orientation = 8
    """
    patches = []

    nrow = im.shape[0]
    ncol = im.shape[1]
    pollshape = poll.shape

    for pr in range(pollshape[0]):
        for pc in range(pollshape[1]):
            if poll[pr,pc]>threshold:
                
                patch = TPatch.TPatch()
                center = ((pr+1)*sampRate - 1,(pc+1)*sampRate - 1,zslice)
                ru = max(center[0] - patch_size,0)
                rd = min(center[0] + patch_size,nrow-1)
                cl = max(center[1] - patch_size,0)
                cr = min(center[1] + patch_size,ncol-1)               
                pdata = im[ru:rd,cl:cr]
                
                patch.image_center = center
                patch.patch_center = (center[0] - ru,center[1] - cl)
                patch.pdata = pdata
                patch.data_size = pdata.shape
                patch.patch_size = patch_size
                patches.append(patch)

    return patches





