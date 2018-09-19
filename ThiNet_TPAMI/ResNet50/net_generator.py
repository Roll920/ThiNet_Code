import sys
caffe_root = '/home/luojh2/Software/caffe-master/'
sys.path.insert(0, caffe_root + 'python')
from caffe.proto import caffe_pb2
import os.path as osp
import os
import numpy as np


class Solver:

    def __init__(self, solver_name=None, folder=None, b=0, compress_block=0):
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
        self.p.test_iter.extend([5000])
        self.p.test_interval = 10000
        self.p.test_initialization = True
        
        if compress_block==0:
            self.p.base_lr = 0.0001
        else:
            self.p.base_lr = 0.0001

        if b == 15 and compress_block==1:  # the final layer
            self.p.base_lr = 0.001
            self.p.max_iter = 90000
            self.p.stepvalue.extend([30000, 60000])
        else:
            self.p.max_iter = 10000

        self.p.lr_policy = self.policy.multistep
        self.p.gamma = 0.1
        self.p.momentum = 0.9
        self.p.weight_decay = 0.0005
        self.p.display = 20

        self.p.snapshot = 10000
        self.p.snapshot_prefix = osp.join(self.folder, "snapshot/")
        self.p.solver_mode = self.machine.GPU

        # self.p.type = self.method.nesterov
        self.p.net = osp.join(self.folder, "trainval.prototxt")

    def write(self):
        if not osp.exists(self.p.snapshot_prefix):
            os.mkdir(self.p.snapshot_prefix)
        with open(self.name, 'wb') as f:
            f.write(str(self.p))


class Net:

    def __init__(self, name="network"):
        self.net = caffe_pb2.NetParameter()
        self.net.name = name
        self.bottom = None
        self.cur = None
        self.this = None

    def setup(self, name, layer_type, bottom=[], top=[], inplace=False):

        self.bottom = self.cur

        new_layer = self.net.layer.add()

        new_layer.name = name
        new_layer.type = layer_type

        if self.bottom is not None and new_layer.type != 'Data':
            bottom_name = [self.bottom.name]
            if len(bottom) == 0:
                bottom = bottom_name
            new_layer.bottom.extend(bottom)

        if inplace:
            top = bottom_name
        elif len(top) == 0:
            top = [name]
        new_layer.top.extend(top)

        self.this = new_layer
        if not inplace:
            self.cur = new_layer

    def suffix(self, name, self_name=None):
        if self_name is None:
            return self.cur.name + '_' + name
        else:
            return self_name

    def write(self, name=None, folder=None, deploy=False):
        # dirname = osp.dirname(name)
        # if not osp.exists(dirname):
        #     os.mkdir(dirname)
        if folder is not None:
            name = osp.join(folder, name)
        elif name is None:
            name = 'trainval.pt'
        else:
            filepath, ext = osp.splitext(name)
            if ext == '':
                ext = '.prototxt'
                name = filepath + ext
        with open(name, 'wb') as f:
            if deploy:
                f.write('name: "resnet-50"\n')
                f.write('input: "data"\ninput_dim: 1\ninput_dim: 3\n')
                f.write('input_dim: 224\ninput_dim: 224\n')
                net_str = str(self.net)
                net_str = net_str[85:-1]
                f.write(net_str)
            else:
                f.write(str(self.net))

    def show(self):
        print self.net
    #************************** params **************************

    def param(self, lr_mult=1, decay_mult=0):
        new_param = self.this.param.add()
        new_param.lr_mult = lr_mult
        new_param.decay_mult = decay_mult

    def transform_param(self, mean_value=128, batch_size=128, scale=.0078125, mirror=1, crop_size=None,
                        mean_file_size=None, phase=None):
        new_transform_param = self.this.transform_param
        new_transform_param.mean_value.extend([104])
        new_transform_param.mean_value.extend([117])
        new_transform_param.mean_value.extend([123])

        new_transform_param.mirror = mirror
        if crop_size is not None:
            new_transform_param.crop_size = crop_size

    def data_param(self, source, backend='LMDB', batch_size=30):
        new_data_param = self.this.data_param
        new_data_param.source = source
        if backend == 'LEVELDB':
            new_data_param.backend = new_data_param.LEVELDB
        else:
            new_data_param.backend = new_data_param.LMDB
        new_data_param.batch_size = batch_size

    def weight_filler(self, filler='msra'):
        """xavier"""
        if self.this.type == 'InnerProduct':
            self.this.inner_product_param.weight_filler.type = filler
        else:
            self.this.convolution_param.weight_filler.type = filler

    def bias_filler(self, filler='constant', value=0):
        if self.this.type == 'InnerProduct':
            self.this.inner_product_param.bias_filler.type = filler
            self.this.inner_product_param.bias_filler.value = value
        else:
            self.this.convolution_param.bias_filler.type = filler
            self.this.convolution_param.bias_filler.value = value

    def include(self, phase='TRAIN'):
        if phase is not None:
            includes = self.this.include.add()
            if phase == 'TRAIN':
                includes.phase = caffe_pb2.TRAIN
            elif phase == 'TEST':
                includes.phase = caffe_pb2.TEST
        else:
            NotImplementedError

    #************************** inplace **************************
    def ReLU(self, name=None):

        self.setup(self.suffix('relu', name), 'ReLU', inplace=True)

    def BatchNorm(self, name=None):

        self.setup(self.suffix('bn', name), 'BatchNorm', inplace=False)
        # self.this.batch_norm_param.use_global_stats = True
        batch_norm_param = self.this.batch_norm_param

    def Scale(self, name=None):
        self.setup(self.suffix('scale', name), 'Scale', inplace=True)
        self.this.scale_param.bias_term = True

    #************************** layers **************************

    def Data(self, source, backend, top=['data', 'label'], name="data", batch_size=30, phase=None, **kwargs):
        self.setup(name, 'Data', top=top)

        self.include(phase)

        self.data_param(source, batch_size=batch_size, backend=backend)
        self.transform_param(phase=phase, **kwargs)

    def Convolution(self, name, bottom=[], num_output=None, kernel_size=3, pad=1, stride=1, decay=True, bias=False, freeze=False, bias_term=None):
        self.setup(name, 'Convolution', bottom=bottom, top=[name])

        conv_param = self.this.convolution_param
        if num_output is None:
            num_output = self.bottom.convolution_param.num_output

        conv_param.num_output = num_output
        conv_param.kernel_size.extend([kernel_size])
        conv_param.pad.extend([pad])
        conv_param.stride.extend([stride])

        if freeze:
            lr_mult = 0
        else:
            lr_mult = 1
        if decay:
            decay_mult = 1
        else:
            decay_mult = 0
        self.param(lr_mult=lr_mult, decay_mult=decay_mult)
        self.weight_filler()

        if bias:
            if decay:
                decay_mult = 2
            else:
                decay_mult = 0
            self.param(lr_mult=lr_mult, decay_mult=decay_mult)
            self.bias_filler()
        if bias_term is not None:
            conv_param.bias_term = bias_term

    def SoftmaxWithLoss(self, name='loss', label='label'):
        self.setup(name, 'SoftmaxWithLoss', bottom=[self.cur.name, label])

    def Softmax(self, bottom=[], name='softmax'):
        self.setup(name, 'Softmax', bottom=bottom)

    def Accuracy(self, name='Accuracy', label='label'):
        cur_bottom = 'fc1000'
        self.setup(name + '/top1', 'Accuracy',
                   bottom=[cur_bottom, label], top=['accuracy@1'])
        self.this.accuracy_param.top_k = 1
        self.setup(name + '/top5', 'Accuracy',
                   bottom=[cur_bottom, label], top=['accuracy@5'])
        self.this.accuracy_param.top_k = 5

    def InnerProduct(self, name='fc', num_output=10):
        self.setup(name, 'InnerProduct')
        self.param(lr_mult=1, decay_mult=1)
        self.param(lr_mult=2, decay_mult=0)
        inner_product_param = self.this.inner_product_param
        inner_product_param.num_output = num_output
        self.weight_filler()
        self.bias_filler()

    def Pooling(self, name, pool='AVE', global_pooling=False, kernel_size=3, stride=2):
        """MAX AVE """
        self.setup(name, 'Pooling')
        if pool == 'AVE':
            self.this.pooling_param.pool = self.this.pooling_param.AVE
        else:
            self.this.pooling_param.pool = self.this.pooling_param.MAX
        if global_pooling:
            self.this.pooling_param.global_pooling = global_pooling
        else:
            self.this.pooling_param.kernel_size = kernel_size
            self.this.pooling_param.stride = stride

    def Eltwise(self, name, bottom1, operation='SUM'):
        bottom0 = self.bottom.name
        self.setup(name, 'Eltwise', bottom=[bottom0, bottom1])

    #************************** DIY **************************
    def conv_relu(self, name, relu_name=None, **kwargs):
        self.Convolution(name, **kwargs)
        self.ReLU(relu_name)

    def conv_bn_relu(self, name, bn_name=None, scale_name=None, relu_name=None, **kwargs):
        self.Convolution(name, **kwargs)
        self.BatchNorm(bn_name)
        self.Scale(scale_name)
        self.ReLU(relu_name)

    def conv_bn(self, name, bn_name=None, scale_name=None, **kwargs):
        self.Convolution(name, **kwargs)
        self.BatchNorm(bn_name)
        self.Scale(scale_name)

    def softmax_acc(self, bottom, **kwargs):
        self.Softmax(bottom=[bottom])

        has_label = None
        for name, value in kwargs.items():
            if name == 'label':
                has_label = value
        if has_label is None:
            self.Accuracy()
        else:
            self.Accuracy(label=has_label)

    #************************** network blocks **************************
    def res_block(self, name, rate1, rate2, num_output, proj=False, p_stride=1):
        bottom = self.cur.name

        self.conv_bn_relu('res' + name + '_branch2a', bn_name='bn' + name + '_branch2a',
                          scale_name='scale' + name + '_branch2a', relu_name='res' + name + '_branch2a' + '_relu',
                          num_output=int(rate1 * num_output), kernel_size=1, pad=0, stride=p_stride, bias_term=False)
        self.conv_bn_relu('res' + name + '_branch2b', bn_name='bn' + name + '_branch2b',
                          scale_name='scale' + name + '_branch2b', relu_name='res' + name + '_branch2b' + '_relu',
                          num_output=int(rate2 * num_output), kernel_size=3, pad=1, stride=1, bias_term=False)
        self.conv_bn('res' + name + '_branch2c', bn_name='bn' + name + '_branch2c',
                     scale_name='scale' + name + '_branch2c',
                     num_output=4 * num_output, kernel_size=1, pad=0, stride=1, bias_term=False)

        if proj:
            self.conv_bn('res' + name + '_branch1', bn_name='bn' + name + '_branch1',
                         scale_name='scale' + name + '_branch1',
                         num_output=4 * num_output, bottom=[bottom], kernel_size=1, pad=0, stride=p_stride, bias_term=False)
            # Important modify!!
            self.Eltwise('res' + name, bottom1='bn' + name + '_branch2c')
            self.setup('res' + name + '_relu', 'ReLU', inplace=True)
        else:
            self.Eltwise('res' + name, bottom1=bottom)
            self.setup('res' + name + '_relu', 'ReLU', inplace=True)

    #************************** networks **************************
    def resnet_50(self, layers, compress_layer, compress_rate, compress_block, deploy=False):
        self.conv_bn_relu('conv1', bn_name='bn_conv1', scale_name='scale_conv1', relu_name='conv1_relu',
                          num_output=64, kernel_size=7, pad=3, stride=2)
        self.Pooling("pool1", pool='max', global_pooling=False,
                     kernel_size=3, stride=2)

        if compress_block == 0:
            # compress 1, not 2
            rate1 = np.ones([1, 16])
            for i in range(0, compress_layer+1):
                rate1[0, i] = rate1[0, i] * compress_rate
            rate2 = np.ones([1, 16])
            for i in range(0, compress_layer):
                rate2[0, i] = rate2[0, i] * compress_rate
        else:
            # both
            rate1 = np.ones([1, 16])
            for i in range(0, compress_layer + 1):
                rate1[0, i] = rate1[0, i] * compress_rate
            rate2 = rate1.copy()

        # 2
        output = 64
        self.res_block(layers[0], rate1[0, 0], rate2[0, 0], output, proj=True)
        self.res_block(layers[1], rate1[0, 1], rate2[0, 1], output, proj=False)
        self.res_block(layers[2], rate1[0, 2], rate2[0, 2], output, proj=False)
        # 3
        output = 128
        self.res_block(layers[3], rate1[0, 3], rate2[0, 3], output, proj=True, p_stride=2)
        self.res_block(layers[4], rate1[0, 4], rate2[0, 4], output, proj=False)
        self.res_block(layers[5], rate1[0, 5], rate2[0, 5], output, proj=False)
        self.res_block(layers[6], rate1[0, 6], rate2[0, 6], output, proj=False)
        # 4
        output = 256
        self.res_block(layers[7], rate1[0, 7], rate2[0, 7], output, proj=True, p_stride=2)
        self.res_block(layers[8], rate1[0, 8], rate2[0, 8], output, proj=False)
        self.res_block(layers[9], rate1[0, 9], rate2[0, 9], output, proj=False)
        self.res_block(layers[10], rate1[0, 10], rate2[0, 10], output, proj=False)
        self.res_block(layers[11], rate1[0, 11], rate2[0, 11], output, proj=False)
        self.res_block(layers[12], rate1[0, 12], rate2[0, 12], output, proj=False)
        # 5
        output = 512
        self.res_block(layers[13], rate1[0, 13], rate2[0, 13], output, proj=True, p_stride=2)
        self.res_block(layers[14], rate1[0, 14], rate2[0, 14], output, proj=False)
        self.res_block(layers[15], rate1[0, 15], rate2[0, 15], output, proj=False)

        self.Pooling("pool5", pool='AVE', global_pooling=False,
                     kernel_size=7, stride=1)
        self.InnerProduct(name='fc1000', num_output=1000)
        if deploy:
            self.Softmax()
        else:
            self.SoftmaxWithLoss()
            self.Accuracy()


def solver_and_prototxt(compress_layer, compress_rate, compress_block):
    layers = ['2a', '2b', '2c', '3a', '3b', '3c', '3d',
              '4a', '4b', '4c', '4d', '4e', '4f', '5a', '5b', '5c']
    pt_folder = layers[compress_layer] + '_' + str(compress_block)
    if not os.path.exists(pt_folder):
        os.mkdir(pt_folder)
    name = 'resnet-' + layers[compress_layer] + str(compress_block) +'-ImageNet'

    solver = Solver(folder=pt_folder, b=compress_layer, compress_block=compress_block)
    solver.write()

    builder = Net(name)
    builder.Data('/opt/luojh/Dataset/ImageNet/lmdb/ilsvrc12_train_lmdb', backend='LMDB', phase='TRAIN', mirror=True,
                 crop_size=224, batch_size=32)
    builder.Data('/opt/luojh/Dataset/ImageNet/lmdb/ilsvrc12_val_lmdb', backend='LMDB', phase='TEST', mirror=False,
                 crop_size=224, batch_size=10)
    builder.resnet_50(layers, compress_layer, compress_rate, compress_block)
    builder.write(name='trainval.prototxt', folder=pt_folder)

    if compress_block == 0:
        compress_block = 1
        compress_layer -= 1
    else:
        compress_block =0

    builder = Net(name + '-old')
    builder.setup('data', 'Data', top=['data'])
    builder.resnet_50(layers, compress_layer, compress_rate, compress_block, deploy=True)
    builder.write(name='deploy.prototxt', folder=pt_folder, deploy=True)
    print "Finished net prototxt generation!"


if __name__ == '__main__':
    compress_layer = 0
    compress_block = 0
    compress_rate = 0.5

    solver_and_prototxt(compress_layer, compress_rate, compress_block)
