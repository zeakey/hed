# -*- coding: utf-8 -*-
import numpy as np
import scipy.misc
import cv2
import scipy.io
import os, sys, argparse
from os.path import join, splitext, split, isfile
parser = argparse.ArgumentParser(description='Forward all testing images.')
parser.add_argument('--model', type=str, default='snapshot/hed_pretrained_bsds.caffemodel')
parser.add_argument('--net', type=str, default='model/hed_test.pt')
parser.add_argument('--output', type=str, default='sigmoid_fuse') # output field
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--ms', type=bool, default=True) # Using multiscale
parser.add_argument('--savemat', type=bool, default=False) # whether save .mat
args = parser.parse_args()
sys.path.insert(0, 'caffe/python')
import caffe
def forward(data):
  assert data.ndim == 3
  data -= np.array((104.00698793,116.66876762,122.67891434))
  data = data.transpose((2, 0, 1))
  net.blobs['data'].reshape(1, *data.shape)
  net.blobs['data'].data[...] = data
  return net.forward()
assert isfile(args.model) and isfile(args.net), 'file not exists'
caffe.set_mode_gpu()
caffe.set_device(args.gpu)

net = caffe.Net(args.net, args.model, caffe.TEST)
test_dir = 'data/HED-BSDS/test/' # test images directory
save_dir = join('data/edge-results/', splitext(split(args.model)[1])[0]) # directory to save results
if args.ms:
  save_dir = save_dir + '_multiscale'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
imgs = [i for i in os.listdir(test_dir) if '.jpg' in i]
nimgs = len(imgs)
print "totally "+str(nimgs)+"images"
for i in range(nimgs):
  img = imgs[i]
  img = cv2.imread(join(test_dir, img)).astype(np.float32)
  if img.ndim == 2:
    img = img[:, :, np.newaxis]
    img = np.repeat(img, 3, 2)
  h, w, _ = img.shape
  edge = np.zeros((h, w), np.float32)
  if args.ms:
    scales = [0.5, 1, 1.5]
  else:
    scales = [1]
  for s in scales:
    h1, w1 = int(s * h), int(s * w)
    img1 = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    edge1 = np.squeeze(forward(img1)[args.output][0, 0, :, :])
    edge += cv2.resize(edge1, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
  edge /= len(scales)
  fn, ext = splitext(imgs[i])
  if args.savemat:
    scipy.io.savemat(join(save_dir, fn),dict({'edge': edge / edge.max()}),appendmat=True)
  scipy.misc.imsave(join(save_dir, fn+'.png'), edge / edge.max())
  print "Saving to '" + join(save_dir, imgs[i][0:-4]) + "', Processing %d of %d..."%(i + 1, nimgs)

