from __future__ import division
import numpy as np
import sys, os, argparse
from os.path import isfile, join, isdir
sys.path.insert(0, 'model')
sys.path.insert(0, 'lib')
import caffe
parser = argparse.ArgumentParser(description='Training hed.')
parser.add_argument('--gpu', type=int, help='gpu ID', default=0)
parser.add_argument('--solver', type=str, help='solver', default='model/hed_solver.pt')
parser.add_argument('--weights', type=str, help='base model', default='model/vgg16convs.caffemodel')
parser.add_argument('--caffe', type=str, help='base model', default='caffe')
args = parser.parse_args()
sys.path.insert(0, join(args.caffe, 'python'))
assert isfile(args.weights) and isfile(args.solver)
caffe.set_mode_gpu()
caffe.set_device(args.gpu)
if not isdir('snapshot'):
  os.makedirs('snapshot')
solver = caffe.SGDSolver(args.solver)
solver.net.copy_from(args.weights)
for p in solver.net.params:
  param = solver.net.params[p]
  for i in range(len(param)):
    print p, "param[%d]: mean=%.5f, std=%.5f"%(i, solver.net.params[p][i].data.mean(), \
    solver.net.params[p][0].data.mean())
solver.solve()

