
import ctypes
from ctypes import WinDLL
import cv2
import numpy as np
from numpy.ctypeslib import ndpointer

class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int),
                ("y", ctypes.c_int)]

class TPScpp(object):
    
    def __init__(self, pS, pD):
        self.TPScpp = WinDLL('CThinPlateSpline.dll')
        self.TPScpp.new_CThinPlateSpline.argtypes = [[Point],[Point]]
        self.TPScpp.new_CThinPlateSpline.restype = ctypes.c_void_p

        self.TPScpp.warpImage.argtypes = [ndpointer(np.float32, flags="C_CONTIGUOUS"), ndpointer(np.float32, flags="C_CONTIGUOUS")]
        self.TPScpp.warpImage.restype = ctypes.c_void_p

        self.obj = self.TPScpp.new_CThinPlateSpline(pS, pD)

    def warp(self, a, b):
        self.TPScpp.warpImage(self.obj, src, dst)

    def delete(): 
        self.TPScpp.del_CThinPlateSpline()
