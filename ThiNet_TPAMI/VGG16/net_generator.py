import sys
caffe_root = '/home/luojh2/Software/caffe-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import numpy as np
import os.path as osp
import os


class Solver:

    def __init__(self, solver_name=None, folder=None, b=0):
        self.solver_name = solver_name
        self.folder = folder

        if self.folder is not None:
            self.name = osp.join(self.folder, 'solver.prototxt')
        if self.name is None:
            self.name = 'solver.pt'
        else:
            filepath, ext = osp.splitext(self.name)
            if ext == '':
                ext = '.prototxt'
                self.name = filepath + ext

        self.p = caffe_pb2.SolverParameter()

        class Method:
            nesterov = "Nesterov"
            SGD = "SGD"
            AdaGrad = "AdaGrad"
            RMSProp = "RMSProp"
            AdaDelta = "AdaDelta"
            Adam = "Adam"
        self.method = Method()

        class Policy:
            """    - fixed: always return base_lr."""
            fixed = 'fixed'
            """    - step: return base_lr * gamma ^ (floor(iter / step))"""
            """    - exp: return base_lr * gamma ^ iter"""
            """    - inv: return base_lr * (1 + gamma * iter) ^ (- power)"""
            """    - multistep: similar to step but it allows non uniform steps defined by stepvalue"""
            multistep = 'multistep'
            """    - poly: the effective learning rate follows a polynomial decay, to be zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)"""
            """    - sigmoid: the effective learning rate follows a sigmod decay"""
            """      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))"""
        self.policy = Policy()

        class Machine:
            GPU = self.p.GPU
            CPU = self.p.GPU
        self.machine = Machine()

        # defaults
        self.p.test_iter.extend([1000])
        self.p.test_interval = 10000
        self.p.test_initialization = True
        if b == 9:  # the final layer
            self.p.max_iter = 120000
            self.p.stepvalue.extend([40000, 80000])
        elif b in [1, 3, 6]:
            self.p.max_iter = 20000
            self.p.stepvalue.extend([10000])
        else:
            self.p.max_iter = 10000
        self.p.base_lr = 0.001
        self.p.lr_policy = self.policy.multistep

        self.p.gamma = 0.1
        self.p.momentum = 0.9
        self.p.weight_decay = 0.0005
        self.p.display = 20

        self.p.snapshot = 10000
        self.p.snapshot_prefix = osp.join(self.folder, "snapshot/")
        self.p.solver_mode = self.machine.GPU

        self.p.net = osp.join(self.folder, "trainval.prototxt")

    def write(self):
        if not osp.exists(self.p.snapshot_prefix):
            os.mkdir(self.p.snapshot_prefix)
        with open(self.name, 'wb') as f:
            f.write(str(self.p))


# helper functions for common structures
def conv_relu(bottom, ks, nout, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, num_output=nout, pad=pad, param=[
                         dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=False)


def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout, param=[
        dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return fc, L.ReLU(fc, in_place=False)


def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def ave_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=ks, stride=stride)


def vgg_16(lmdb, bs_train=16, bs_val=50, rate=0, lmdb_flag=False, not_deploy=True):
    n = caffe.NetSpec()
    if not_deploy:
        if lmdb_flag:
            n.data, n.label = L.Data(source=lmdb + 'ilsvrc12_train_lmdb', backend=P.Data.LMDB,
                                     include=dict(phase=caffe_pb2.TRAIN), batch_size=bs_train, ntop=2,
                                     transform_param=dict(crop_size=224, mean_value=[104, 117, 123], mirror=True))
            data_str = n.to_proto()
            n.data, n.label = L.Data(source=lmdb + 'ilsvrc12_val_lmdb', backend=P.Data.LMDB,
                                     include=dict(phase=caffe_pb2.TEST), batch_size=bs_val, ntop=2,
                                     transform_param=dict(crop_size=224, mean_value=[104, 117, 123], mirror=False))
        else:
            n.data, n.label = L.Data(source=lmdb + 'ilsvrc12_train_leveldb', backend=P.Data.LEVELDB,
                                     include=dict(phase=caffe_pb2.TRAIN), batch_size=bs_train, ntop=2,
                                     transform_param=dict(crop_size=224, mean_value=[104, 117, 123], mirror=True))
            data_str = n.to_proto()
            n.data, n.label = L.Data(source=lmdb + 'ilsvrc12_val_leveldb', backend=P.Data.LEVELDB,
                                     include=dict(phase=caffe_pb2.TEST), batch_size=bs_val, ntop=2,
                                     transform_param=dict(crop_size=224, mean_value=[104, 117, 123], mirror=False))
    else:
        data_str = 'input: "data"\ninput_dim: 1\ninput_dim: 3\ninput_dim: 224\ninput_dim: 224'
        n.data = L.Data()

    # the net itself
    n.conv1_1, n.relu1_1 = conv_relu(
        n.data, nout=int(rate[0] * 64), pad=1, ks=3)
    n.conv1_2, n.relu1_2 = conv_relu(
        n.relu1_1, nout=int(rate[1] * 64), pad=1, ks=3)
    n.pool1 = max_pool(n.relu1_2, ks=2, stride=2)

    n.conv2_1, n.relu2_1 = conv_relu(
        n.pool1, nout=int(rate[2] * 128), pad=1, ks=3)
    n.conv2_2, n.relu2_2 = conv_relu(
        n.relu2_1, nout=int(rate[3] * 128), pad=1, ks=3)
    n.pool2 = max_pool(n.relu2_2, ks=2, stride=2)

    n.conv3_1, n.relu3_1 = conv_relu(
        n.pool2, nout=int(rate[4] * 256), pad=1, ks=3)
    n.conv3_2, n.relu3_2 = conv_relu(
        n.relu3_1, nout=int(rate[5] * 256), pad=1, ks=3)
    n.conv3_3, n.relu3_3 = conv_relu(
        n.relu3_2, nout=int(rate[6] * 256), pad=1, ks=3)
    n.pool3 = max_pool(n.relu3_3, ks=2, stride=2)

    n.conv4_1, n.relu4_1 = conv_relu(
        n.pool3, nout=int(rate[7] * 512), pad=1, ks=3)
    n.conv4_2, n.relu4_2 = conv_relu(
        n.relu4_1, nout=int(rate[8] * 512), pad=1, ks=3)
    n.conv4_3, n.relu4_3 = conv_relu(
        n.relu4_2, nout=int(rate[9] * 512), pad=1, ks=3)
    n.pool4 = max_pool(n.relu4_3, ks=2, stride=2)

    n.conv5_1, n.relu5_1 = conv_relu(
        n.pool4, nout=int(rate[10] * 512), pad=1, ks=3)
    n.conv5_2, n.relu5_2 = conv_relu(
        n.relu5_1, nout=int(rate[11] * 512), pad=1, ks=3)
    n.conv5_3, n.relu5_3 = conv_relu(
        n.relu5_2, nout=int(rate[12] * 512), pad=1, ks=3)
    n.pool5 = max_pool(n.relu5_3, ks=2, stride=2)

    n.fc6, n.relu6 = fc_relu(n.pool5, nout=4096)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7, n.relu7 = fc_relu(n.relu6, nout=4096)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

    n.fc8 = L.InnerProduct(n.relu7, num_output=1000, param=[
        dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    if not_deploy:
        n.loss = L.SoftmaxWithLoss(n.fc8, n.label)
        n.acc_top_1 = L.Accuracy(n.fc8, n.label, top_k=1)
        n.acc_top_5 = L.Accuracy(n.fc8, n.label, top_k=5)
    else:
        n.prob = L.Softmax(n.fc8)
    model_str = str(n.to_proto())
    if not not_deploy:
        model_str = model_str[54:-1]
    return str(data_str) + '\n' + model_str


def solver_and_prototxt(compress_layer, compress_rate):
    compress_rate_tmp = np.ones(13)
    compress_rate_old = compress_rate_tmp
    for i in range(compress_layer):
        compress_rate_old[i] = compress_rate
    compress_rate_new = np.array(compress_rate_old)
    compress_rate_new[compress_layer] = compress_rate

    candidate = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1',
                 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']

    pt_folder = osp.join(candidate[compress_layer])
    if not os.path.exists(pt_folder):
        os.mkdir(pt_folder)
    solver = Solver(folder=pt_folder, b=compress_layer)
    solver.write()

    with open(candidate[compress_layer] + '/trainval.prototxt', 'w') as f:
        f.write(
            vgg_16('/opt/luojh/Dataset/ImageNet/keep/lmdb/', bs_train=32,
                   bs_val=50, rate=compress_rate_new, lmdb_flag=True))

    with open(candidate[compress_layer] + '/deploy.prototxt', 'w') as f:
        f.write(
            vgg_16('/opt/luojh/Dataset/ImageNet/keep/lmdb/',
                   rate=compress_rate_old, lmdb_flag=True, not_deploy=False))


if __name__ == '__main__':
    compress_layer = 9
    compress_rate = 0.4
    solver_and_prototxt(compress_layer, compress_rate)
