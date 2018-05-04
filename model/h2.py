import sys, os, argparse
from os.path import join, isdir, isfile, split
sys.path.insert(0, 'caffe/python')
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop
import numpy as np
from math import ceil
parser = argparse.ArgumentParser(description='Training hed.')
parser.add_argument('--nfeat', type=int, help='number features', default=11)
parser.add_argument('--bias', type=bool, default=True)
args = parser.parse_args()
tmp_dir = 'tmp'
if not isdir(tmp_dir):
  os.makedirs(tmp_dir)

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, mult=[1,1,2,0]):
  conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
    num_output=nout, pad=pad, weight_filler=dict(type='msra'), 
    param=[dict(lr_mult=mult[0], decay_mult=mult[1]), dict(lr_mult=mult[2], decay_mult=mult[3])])
  return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
  return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def conv1x1(bottom, nout=1, lr=[0.01, 1, 0.02, 0], wf=dict(type="constant")):
  if args.bias:
    return L.Convolution(bottom, kernel_size=1, num_output=nout, weight_filler=wf,
        param=[dict(lr_mult=lr[0], decay_mult=lr[1]), dict(lr_mult=lr[2], decay_mult=lr[3])])
  else:
    return L.Convolution(bottom, kernel_size=1, num_output=nout, weight_filler=wf,
        bias_term=False, param=[dict(lr_mult=lr[0], decay_mult=lr[1])])

def upsample(bottom, stride, nout=1, name=None):
  s, k, pad = stride, 2 * stride, int(ceil(stride-1)/2)
  if not name:
    name = "upsample%d"%s
  return L.Deconvolution(bottom, name=name, convolution_param=dict(num_output=nout, bias_term=False,
    kernel_size=k, stride=s, pad=pad, weight_filler = dict(type="bilinear"), group=nout), 
    param=[dict(lr_mult=0, decay_mult=0)])

def net(split):
  n = caffe.NetSpec()
  # loss_param = dict(normalization=P.Loss.VALID)
  loss_param = dict(normalize=False)
  if split=='train':
    data_params = dict(mean=(104.00699, 116.66877, 122.67892))
    data_params['root'] = 'data/HED-BSDS_PASCAL'
    data_params['source'] = "bsds_pascal_train_pair.lst"
    data_params['shuffle'] = True
    data_params['ignore_label'] = -1
    n.data, n.label = L.Python(module='pylayer', layer='ImageLabelmapDataLayer', ntop=2, \
    param_str=str(data_params))
    if data_params.has_key('ignore_label'):
      loss_param['ignore_label'] = int(data_params['ignore_label'])
  elif split == 'test':
    n.data = L.Input(name = 'data', input_param=dict(shape=dict(dim=[1,3,200,200])))
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
  ## w1
  n.w1_1top = conv1x1(n.conv1_1, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.w1_2top = conv1x1(n.conv1_2, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  ## w2
  n.w2_1top = conv1x1(n.conv2_1, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.w2_2top = conv1x1(n.conv2_2, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.w2_1down = conv1x1(n.conv2_1, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.w2_2down = conv1x1(n.conv2_2, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  ## w3
  n.w3_1top = conv1x1(n.conv3_1, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.w3_2top = conv1x1(n.conv3_2, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.w3_3top = conv1x1(n.conv3_3, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.w3_1down = conv1x1(n.conv3_1, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.w3_2down = conv1x1(n.conv3_2, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.w3_3down = conv1x1(n.conv3_3, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  ## w4
  n.w4_1top = conv1x1(n.conv4_1, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.w4_2top = conv1x1(n.conv4_2, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.w4_3top = conv1x1(n.conv4_3, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.w4_1down = conv1x1(n.conv4_1, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.w4_2down = conv1x1(n.conv4_2, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.w4_3down = conv1x1(n.conv4_3, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  ## w5
  n.w5_1down = conv1x1(n.conv5_1, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.w5_2down = conv1x1(n.conv5_2, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.w5_3down = conv1x1(n.conv5_3, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))

  ## upsample wx_xdown
  n.w2_1down_up = upsample(n.w2_1down, nout=args.nfeat, stride=2, name='upsample2_1')
  n.w2_2down_up = upsample(n.w2_2down, nout=args.nfeat, stride=2, name='upsample2_2')
  
  n.w3_1down_up = upsample(n.w3_1down, nout=args.nfeat, stride=2, name='upsample3_1')
  n.w3_2down_up = upsample(n.w3_2down, nout=args.nfeat, stride=2, name='upsample3_2')
  n.w3_3down_up = upsample(n.w3_3down, nout=args.nfeat, stride=2, name='upsample3_3')
  
  n.w4_1down_up = upsample(n.w4_1down, nout=args.nfeat, stride=2, name='upsample4_1')
  n.w4_2down_up = upsample(n.w4_2down, nout=args.nfeat, stride=2, name='upsample4_2')
  n.w4_3down_up = upsample(n.w4_3down, nout=args.nfeat, stride=2, name='upsample4_3')
  
  n.w5_1down_up = upsample(n.w5_1down, nout=args.nfeat, stride=2, name='upsample5_1')
  n.w5_2down_up = upsample(n.w5_2down, nout=args.nfeat, stride=2, name='upsample5_2')
  n.w5_3down_up = upsample(n.w5_3down, nout=args.nfeat, stride=2, name='upsample5_3')
  
  ## crop wx_xdown_up
  n.w2_1down_up_crop = crop(n.w2_1down_up, n.w1_1top)
  n.w2_2down_up_crop = crop(n.w2_2down_up, n.w1_1top)
  
  n.w3_1down_up_crop = crop(n.w3_1down_up, n.w2_1top)
  n.w3_2down_up_crop = crop(n.w3_2down_up, n.w2_1top)
  n.w3_3down_up_crop = crop(n.w3_3down_up, n.w2_1top)
  
  n.w4_1down_up_crop = crop(n.w4_1down_up, n.w3_1top)
  n.w4_2down_up_crop = crop(n.w4_2down_up, n.w3_1top)
  n.w4_3down_up_crop = crop(n.w4_3down_up, n.w3_1top)
  
  n.w5_1down_up_crop = crop(n.w5_1down_up, n.w4_1top)
  n.w5_2down_up_crop = crop(n.w5_2down_up, n.w4_1top)
  n.w5_3down_up_crop = crop(n.w5_3down_up, n.w4_1top)

  ## fuse
  n.h1s1_2 = L.Eltwise(n.w1_1top, n.w1_2top, n.w2_1down_up_crop, n.w2_2down_up_crop)
  
  n.h1s2_3 = L.Eltwise(n.w2_1top, n.w2_2top, n.w3_1down_up_crop, n.w3_2down_up_crop, n.w3_3down_up_crop)
  
  n.h1s3_4 = L.Eltwise(n.w3_1top, n.w3_2top, n.w3_3top, \
  	         n.w4_1down_up_crop, n.w4_2down_up_crop, n.w4_3down_up_crop)
  
  n.h1s4_5 = L.Eltwise(n.w4_1top, n.w4_2top, n.w4_3top, \
  	         n.w5_1down_up_crop, n.w5_2down_up_crop, n.w5_3down_up_crop)
  
  ## score h1sx_x
  n.score_h1s1_2 = conv1x1(n.h1s1_2, lr=[0.01, 1, 0.02, 0], wf=dict(type='gaussian', std=0.001))
  n.score_h1s2_3 = conv1x1(n.h1s2_3, lr=[0.01, 1, 0.02, 0], wf=dict(type='gaussian', std=0.001))
  n.score_h1s3_4 = conv1x1(n.h1s3_4, lr=[0.01, 1, 0.02, 0], wf=dict(type='gaussian', std=0.001))
  n.score_h1s4_5 = conv1x1(n.h1s4_5, lr=[0.01, 1, 0.02, 0], wf=dict(type='gaussian', std=0.001))
  ## upsample score
  n.upscore_h1s2_3 = upsample(n.score_h1s2_3, stride=2, name='upscore_h1s2_3')
  n.upscore_h1s3_4 = upsample(n.score_h1s3_4, stride=4, name='upscore_h1s2_4')
  n.upscore_h1s4_5 = upsample(n.score_h1s4_5, stride=8, name='upscore_h1s4_5')
  ## crop upscore_h1sx_x
  n.crop_h1s1_2 = crop(n.score_h1s1_2, n.data)
  n.crop_h1s2_3 = crop(n.upscore_h1s2_3, n.data)
  n.crop_h1s3_4 = crop(n.upscore_h1s3_4, n.data)
  n.crop_h1s4_5 = crop(n.upscore_h1s4_5, n.data)
  ## fuse
  n.h1_concat = L.Concat(n.crop_h1s1_2,
                      n.crop_h1s2_3,
                      n.crop_h1s3_4,
                      n.crop_h1s4_5,
                      concat_param=dict({'concat_dim':1}))
  n.h1_fuse = conv1x1(n.h1_concat, lr=[0.001, 1, 0.002, 0], wf=dict(type='constant', value=float(1)/4))
  if split == 'train':
  	n.loss_h1s1_2 = L.BalanceCrossEntropyLoss(n.crop_h1s1_2, n.label, loss_param=loss_param)
  	n.loss_h1s2_3 = L.BalanceCrossEntropyLoss(n.crop_h1s2_3, n.label, loss_param=loss_param)
  	n.loss_h1s3_4 = L.BalanceCrossEntropyLoss(n.crop_h1s3_4, n.label, loss_param=loss_param)
  	n.loss_h1s4_5 = L.BalanceCrossEntropyLoss(n.crop_h1s4_5, n.label, loss_param=loss_param)
  	n.loss_h1_fuse = L.BalanceCrossEntropyLoss(n.h1_fuse, n.label, loss_param=loss_param)
  else:
  	n.sigmoid_h1s1_2 = L.Sigmoid(n.crop_h1s1_2)
  	n.sigmoid_h1s2_3 = L.Sigmoid(n.crop_h1s2_3)
  	n.sigmoid_h1s3_4 = L.Sigmoid(n.crop_h1s3_4)
  	n.sigmoid_h1s4_5 = L.Sigmoid(n.crop_h1s4_5)
  	n.sigmoid_h1_fuse = L.Sigmoid(n.h1_fuse)
  ## H2: conv h1sx_x for H2 fusing
  n.h1s1_2top  = conv1x1(n.h1s1_2, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.h1s2_3top  = conv1x1(n.h1s2_3, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.h1s2_3down = conv1x1(n.h1s2_3, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.h1s3_4top  = conv1x1(n.h1s3_4, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.h1s3_4down = conv1x1(n.h1s3_4, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  n.h1s4_5down = conv1x1(n.h1s4_5, nout=args.nfeat, lr=[0.1, 1, 0.2, 0], wf=dict(type='gaussian', std=0.001))
  ## upsample H2
  n.h1s2_3upsample = upsample(n.h1s2_3down, nout=args.nfeat, stride=2, name='upsample_h1s2_3')
  n.h1s3_4upsample = upsample(n.h1s3_4down, nout=args.nfeat, stride=2, name='upsample_h1s3_4')
  n.h1s4_5upsample = upsample(n.h1s4_5down, nout=args.nfeat, stride=2, name='upsample_h1s4_5')
  ## Crop H2
  n.h1s2_3crop = crop(n.h1s2_3upsample, n.h1s1_2top)
  n.h1s3_4crop = crop(n.h1s3_4upsample, n.h1s2_3top)
  n.h1s4_5crop = crop(n.h1s4_5upsample, n.h1s3_4top)
  ## fuse H2
  n.h2s1_2_3 = L.Eltwise(n.h1s1_2top, n.h1s2_3crop)
  n.h2s2_3_4 = L.Eltwise(n.h1s2_3top, n.h1s3_4crop)
  n.h2s3_4_5 = L.Eltwise(n.h1s3_4top, n.h1s4_5crop)
  ## score H2
  n.score_h2s1_2_3 = conv1x1(n.h2s1_2_3, lr=[0.01, 1, 0.02, 0], wf=dict(type='gaussian', std=0.001))
  n.score_h2s2_3_4 = conv1x1(n.h2s2_3_4, lr=[0.01, 1, 0.02, 0], wf=dict(type='gaussian', std=0.001))
  n.score_h2s3_4_5 = conv1x1(n.h2s3_4_5, lr=[0.01, 1, 0.02, 0], wf=dict(type='gaussian', std=0.001))
  ## upsample H2 score
  n.score_h2s2_3_4upsample = upsample(n.score_h2s2_3_4, stride=2, name='upscore_h2s2_3_4')
  n.score_h2s3_4_5upsample = upsample(n.score_h2s3_4_5, stride=4, name='upscore_h2s3_4_5')
  ## Crop H2 score
  n.score_h2s1_2_3crop = crop(n.score_h2s1_2_3, n.data)
  n.score_h2s2_3_4crop = crop(n.score_h2s2_3_4upsample, n.data)
  n.score_h2s3_4_5crop = crop(n.score_h2s3_4_5upsample, n.data)
  # concat H2
  n.h2_concat = L.Concat(n.score_h2s1_2_3crop, n.score_h2s2_3_4crop, n.score_h2s3_4_5crop,\
                        concat_param=dict({'concat_dim':1}))
  n.h2_fuse = conv1x1(n.h2_concat, lr=[0.001, 1, 0.002, 0], wf=dict(type='constant', value=0.333))
  if split == 'train':
  	n.loss_h2s1_2_3 = L.BalanceCrossEntropyLoss(n.score_h2s1_2_3crop, n.label, loss_param=loss_param)
  	n.loss_h2s2_3_4 = L.BalanceCrossEntropyLoss(n.score_h2s2_3_4crop, n.label, loss_param=loss_param)
  	n.loss_h2s3_4_5 = L.BalanceCrossEntropyLoss(n.score_h2s3_4_5crop, n.label, loss_param=loss_param)
  	n.loss_h2_fuse = L.BalanceCrossEntropyLoss(n.h2_fuse, n.label, loss_param=loss_param)
  else:
  	n.sigmoid_h2s1_2_3 = L.Sigmoid(n.score_h2s1_2_3crop)
  	n.sigmoid_h2s2_3_4 = L.Sigmoid(n.score_h2s2_3_4crop)
  	n.sigmoid_h2s3_4_5 = L.Sigmoid(n.score_h2s3_4_5crop)
  	n.sigmoid_h2_fuse = L.Sigmoid(n.h2_fuse)
  # Concat H1 and H2
  n.h1h2_concat = L.Concat(n.score_h2s1_2_3crop, n.score_h2s2_3_4crop, n.score_h2s3_4_5crop,
                           n.crop_h1s1_2, n.crop_h1s2_3, n.crop_h1s3_4, n.crop_h1s4_5,
                           concat_param=dict({'concat_dim': 1}))
  n.h1h2_fuse = conv1x1(n.h1h2_concat, lr=[0.001, 1, 0.002, 0], wf=dict(type='constant', value=float(1)/7))
  if split == 'train':
    n.loss_h1h2_fuse = L.BalanceCrossEntropyLoss(n.h1h2_fuse, n.label, loss_param=loss_param)
  else:
    n.sigmoid_h1h2_fuse = L.Sigmoid(n.h1h2_fuse)
  return n.to_proto()

def make_net():
  fpath = join(tmp_dir, "h2feat%d_train.pt"%args.nfeat)
  with open(fpath, 'w') as f:
    f.write(str(net('train')))
  fpath = join(tmp_dir, "h2feat%d_test.pt"%args.nfeat)
  with open(fpath, 'w') as f:
    f.write(str(net('test')))
def make_solver():
  sp = {}
  fpath = join(tmp_dir, "h1feat%d_train.pt"%args.nfeat)
  sp['net'] = '"' + fpath + '"'
  sp['base_lr'] = '0.000001'
  sp['lr_policy'] = '"step"'
  sp['momentum'] = '0.9'
  sp['weight_decay'] = '0.0002'
  sp['iter_size'] = '10'
  sp['stepsize'] = '20000'
  sp['display'] = '10'
  sp['snapshot'] = '2000'
  sp['snapshot_prefix'] = '"snapshot/h2feat%d"'%args.nfeat
  sp['gamma'] = '0.1'
  sp['max_iter'] = '40000'
  sp['solver_mode'] = 'GPU'
  fpath = join(tmp_dir, "h2feat%d_solver.pt"%args.nfeat)
  f = open(fpath, 'w')
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
