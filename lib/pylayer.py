#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Code written by KAI ZHAO(http://kaiz.xyz)
import caffe
import numpy as np
from os.path import join, isfile
import random, cv2

class ImageLabelmapDataLayer(caffe.Layer):
  """
  Python data layer
  """
  def setup(self, bottom, top):
    params = eval(self.param_str)
    self.root = params['root']
    self.source = params['source']
    self.shuffle = bool(params['shuffle'])
    self.mean = np.array(params['mean'], dtype=np.float32)
    assert self.mean.size == 1 or self.mean.size == 3, "mean.size != 1 and mean.size != 3"
    if params.has_key('ignore_label'):
      self.ignore_label = np.float32(params['ignore_label'])
    else:
      self.ignore_label = None
    with open(join(self.root, self.source), 'r') as f:
      self.filelist = f.readlines()
    if self.shuffle:
      random.shuffle(self.filelist)
    self.idx = 0
    top[0].reshape(1, 3, 100, 100) # img
    top[1].reshape(1, 1, 100, 100) # lb
  
  def reshape(self, bottom, top):
    """
    Will reshape in forward()
    """

  def forward(self, bottom, top):
    """
    Load data
    """
    [imgfn, lbfn] = self.filelist[self.idx].split()
    [imgfn, lbfn] = join(self.root, imgfn), join(self.root, lbfn)
    assert isfile(imgfn) and isfile(lbfn), "File not exists!"
    img = cv2.imread(imgfn).astype(np.float32)
    lb = cv2.imread(lbfn, 0).astype(np.float32)
    if img.ndim == 2:
      img = img[:,:,np.newaxis]
      img = np.repeat(img, 3, 2)
    img -= self.mean
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, :, :, :]
    assert lb.ndim == 2, "lb.ndim = %d"%lb.ndim
    h, w = lb.shape
    assert img.shape[2] == h and img.shape[3] == w, "Image and GT shape mismatch."
    lb = lb[np.newaxis, np.newaxis, :, :]
    thres = 125
    if self.ignore_label:
      lb[np.logical_and(lb < thres, lb != 0)] = self.ignore_label
      lb[lb >= thres] = 1
    else:
      lb[lb < thres] = 0
      lb[lb != 0] = 1
    if np.count_nonzero(lb) == 0:
      print "Warning: all zero label map!"
    top[0].reshape(1, 3, h, w)
    top[1].reshape(1, 1, h, w)
    top[0].data[...] = img
    top[1].data[...] = lb
    if self.idx == len(self.filelist)-1:
      # we've reached the end, restart.
      print "Restarting data prefetching from start."
      random.shuffle(self.filelist)
      self.idx = 0
    else:
      self.idx = self.idx + 1

  def backward(self, top, propagate_down, bottom):
    """
    Data layer doesn't need back propagate
    """
    pass
