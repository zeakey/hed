import sys, os
sys.path.insert(0, 'caffe/python')
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop
import numpy as np
from math import ceil

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, mult=[1,1,2,0]):
  conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
    num_output=nout, pad=pad, weight_filler=dict(type='xavier'), 
    param=[dict(lr_mult=mult[0], decay_mult=mult[1]), dict(lr_mult=mult[2], decay_mult=mult[3])])
  return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
  return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def conv1x1(bottom, name, lr=1):
  return L.Convolution(bottom, name=name, kernel_size=1,num_output=1,# weight_filler=dict(type='xavier'),
      param=[dict(lr_mult=0.01*lr, decay_mult=1), dict(lr_mult=0.02*lr, decay_mult=0)])

def upsample(bottom, stride):
  s, k, pad = stride, 2 * stride, int(ceil(stride-1)/2)
  name = "upsample%d"%s
  return L.Deconvolution(bottom, name=name, convolution_param=dict(num_output=1, 
    kernel_size=k, stride=s, pad=pad, weight_filler = dict(type="bilinear")),
      param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])

def net(split):
  n = caffe.NetSpec()
  if split=='train':
    data_params = dict(mean=(104.00699, 116.66877, 122.67892))
    data_params['root'] = 'data/HED-BSDS'
    data_params['source'] = "train_pair.lst"
    data_params['shuffle'] = True
    data_params['ignore_label'] = -1 # ignore label
    n.data, n.label = L.Python(module='pylayer', layer='ImageLabelmapDataLayer', ntop=2, \
    param_str=str(data_params))
    loss_param = dict(normalize=False)
    if data_params.has_key('ignore_label'):
      loss_param['ignore_label'] = data_params['ignore_label']
  elif split == 'test':
    n.data = L.Input(name = 'data', input_param=dict(shape=dict(dim=[1,3,500,500])))
  else:
    raise Exception("Invalid phase")
  
  n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=1)
  n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
  n.pool1 = max_pool(n.relu1_2)

  n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
  n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
  n.pool2 = max_pool(n.relu2_2)

  n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
  n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
  n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
  n.pool3 = max_pool(n.relu3_3)

  n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
  n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
  n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
  n.pool4 = max_pool(n.relu4_3)
  
  n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512, mult=[100,1,200,0])
  n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512, mult=[100,1,200,0])
  n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512, mult=[100,1,200,0])
  
  # DSN1
  n.score_dsn1=conv1x1(n.conv1_2, 'score-dsn1', lr=1)
  n.upscore_dsn1 = crop(n.score_dsn1, n.data)
  if split=='train':
    n.loss1 = L.BalanceCrossEntropyLoss(n.upscore_dsn1, n.label, loss_param=loss_param)
  else:
    n.sigmoid_dsn1 = L.Sigmoid(n.upscore_dsn1)
  # DSN2
  n.score_dsn2 = conv1x1(n.conv2_2, 'score-dsn2')
  n.score_dsn2_up = upsample(n.score_dsn2, stride=2)
  n.upscore_dsn2 = crop(n.score_dsn2_up, n.data)
  if split=='train':
    n.loss2 = L.BalanceCrossEntropyLoss(n.upscore_dsn2, n.label, loss_param=loss_param)
  else:
    n.sigmoid_dsn2 = L.Sigmoid(n.upscore_dsn2)
  # DSN3
  n.score_dsn3=conv1x1(n.conv3_3, 'score-dsn3')
  n.score_dsn3_up = upsample(n.score_dsn3, stride=4)
  n.upscore_dsn3 = crop(n.score_dsn3_up, n.data)
  if split=='train':
    n.loss3 = L.BalanceCrossEntropyLoss(n.upscore_dsn3, n.label, loss_param=loss_param)
  else:
    n.sigmoid_dsn3 = L.Sigmoid(n.upscore_dsn3)
  # DSN4
  n.score_dsn4 = conv1x1(n.conv4_3, 'score-dsn4')
  n.score_dsn4_up = upsample(n.score_dsn4, stride=8)
  n.upscore_dsn4 = crop(n.score_dsn4_up, n.data)
  if split=='train':
    n.loss4 = L.BalanceCrossEntropyLoss(n.upscore_dsn4, n.label, loss_param=loss_param)
  else:
    n.sigmoid_dsn4 = L.Sigmoid(n.upscore_dsn4)
  # DSN5
  n.score_dsn5=conv1x1(n.conv5_3, 'score-dsn5')
  n.score_dsn5_up = upsample(n.score_dsn5, stride=16)
  n.upscore_dsn5 = crop(n.score_dsn5_up, n.data)
  if split=='train':
    n.loss5 = L.BalanceCrossEntropyLoss(n.upscore_dsn5, n.label, loss_param=loss_param)
  elif split=='test':
    n.sigmoid_dsn5 = L.Sigmoid(n.upscore_dsn5)
  # concat and fuse
  n.concat_upscore = L.Concat(n.upscore_dsn1,
                      n.upscore_dsn2,
                      n.upscore_dsn3,
                      n.upscore_dsn4,
                      n.upscore_dsn5,
                      name='concat', concat_param=dict({'concat_dim':1}))
  n.upscore_fuse = L.Convolution(n.concat_upscore, name='new-score-weighting', 
                 num_output=1, kernel_size=1,
                 param=[dict(lr_mult=0.001, decay_mult=1), dict(lr_mult=0.002, decay_mult=0)],
                 weight_filler=dict(type='constant', value=0.2))
  if split=='test':
    n.sigmoid_fuse = L.Sigmoid(n.upscore_fuse)
  else:
    n.loss_fuse = L.BalanceCrossEntropyLoss(n.upscore_fuse, n.label, loss_param=loss_param)
  return n.to_proto()

def make_net():
  with open('model/hed_train.pt', 'w') as f:
    f.write(str(net('train')))
  with open('model/hed_test.pt', 'w') as f:
    f.write(str(net('test')))
def make_solver():
  sp = {}
  sp['net'] = '"model/hed_train.pt"'
  sp['base_lr'] = '0.000001'
  sp['lr_policy'] = '"step"'
  sp['momentum'] = '0.9'
  sp['weight_decay'] = '0.0002'
  sp['iter_size'] = '10'
  sp['stepsize'] = '10000'
  sp['display'] = '10'
  sp['snapshot'] = '2000'
  sp['snapshot_prefix'] = '"snapshot/hed"'
  sp['gamma'] = '0.1'
  sp['max_iter'] = '40000'
  sp['solver_mode'] = 'GPU'
  f = open('model/hed_solver.pt', 'w')
  for k, v in sorted(sp.items()):
      if not(type(v) is str):
          raise TypeError('All solver parameters must be strings')
      f.write('%s: %s\n'%(k, v))
  f.close()

def make_all():
  make_net()
  make_solver()

if __name__ == '__main__':
  make_all()
