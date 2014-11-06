import sys
import pymaxflow
import pylab
import numpy as np
from scipy import misc
import tiffLib
import histEqualization
import AT_denoising

import ImageIO
import TImage

eps = 0.01

dataPath = 'C:/Tomosynthesis/localtest/'
fileName = 'cancer.tif'
outputPath = 'C:/Tomosynthesis/localtest/res/'

im = ImageIO.imReader(dataPath,fileName, 'tif',2)
eqimg = histEqualization.histEqualization(im.data[0], 16)
denoised = AT_denoising.DenoisingAW(eqimg)
denoised = AT_denoising.DenoisingAW(denoised)

tiffLib.imsave(outputPath + 'denoised.tif',denoised)

im = np.float32(denoised)
indices = np.arange(im.size).reshape(im.shape).astype(np.int32)

# create source/sink
sink = indices[im.shape[0]/2 - 20:im.shape[0]/2 + 20,im.shape[1]/2 - 20:im.shape[1]/2 + 20]
source = np.hstack((indices[:, 0],indices[:, -1],indices[0, 1:-1],indices[-1, 1:-1]))

# creat graph 
g = pymaxflow.PyGraph(im.size, im.size * 3)
g.add_node(im.size)

# adjacent horizontal
diffs = np.abs(im[:, 1:] - im[:, :-1]).ravel() + eps

tiffLib.imsave(outputPath + 'adjacenth.tif',np.abs(im[:, 1:] - im[:, :-1]))
e1 = indices[:, :-1].ravel()
e2 = indices[:, 1:].ravel()
g.add_edge_vectorized(e1, e2, 1.0/diffs, 0 * diffs)

# adjacent vertical
diffs = np.abs(im[1:, 0:] - im[:-1, :]).ravel() + eps
tiffLib.imsave(outputPath + 'adjacentv.tif',np.abs(im[1:, 0:] - im[:-1, :]))
e1 = indices[1:, :].ravel()
e2 = indices[:-1, :].ravel()
g.add_edge_vectorized(e1, e2, 1.0/diffs, 0 * diffs)

# adjacent 45
diffs = np.abs(im[1:, 1:] - im[:-1, :-1]).ravel() + eps
tiffLib.imsave(outputPath + 'adjacentup.tif',np.abs(im[1:, 1:] - im[:-1, :-1]))

e1 = indices[1:, :-1].ravel()
e2 = indices[:-1, 1:].ravel()
g.add_edge_vectorized(e1, e2, 1.0/diffs, 0 * diffs)

# adjacent 135
diffs = np.abs(im[:-1, 1:] - im[1:, :-1]).ravel() + eps
tiffLib.imsave(outputPath + 'adjacentdown.tif',np.abs(im[:-1, 1:] - im[1:, :-1]))
e1 = indices[:-1, :-1].flatten()
e2 = indices[1:, 1:].ravel()
g.add_edge_vectorized(e1, e2, 1.0/diffs, 0 * diffs)

# link to source/sink

#g.add_tweights_vectorized(indices[:, 0], (np.ones(indices.shape[0]) * 1.e9).astype(np.float32), np.zeros(indices.shape[0], np.float32))
g.add_tweights_vectorized(source.ravel(), (np.ones(source.size) * 1.e9).astype(np.float32), np.zeros(source.size, np.float32))

#g.add_tweights_vectorized(indices[:, -1], np.zeros(indices.shape[0], np.float32), (np.ones(indices.shape[0]) * 1.e9).astype(np.float32))
g.add_tweights_vectorized(sink.ravel(), np.zeros(sink.size,np.float32), (np.ones(sink.size) * 1.e9).astype(np.float32))

print "calling maxflow"
g.maxflow()

out = g.what_segment_vectorized()
tiffLib.imsave(outputPath  + 'out.tif',np.float32(out.reshape(im.shape)))


