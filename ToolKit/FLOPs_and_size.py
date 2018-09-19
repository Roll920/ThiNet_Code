"""
Copyright 2016 Yahoo Inc.
Licensed under the terms of the 2 clause BSD license.
Please see LICENSE file in the project root for terms.
"""
import sys
import os
if os.path.isfile('../caffe_path.txt'):
    fid = open('../caffe_path.txt', 'r')
    caffe_root = fid.readline().strip('\n')
    fid.close()
else:
    caffe_root = '/home/luojh2/Software/caffe-master/python/'
sys.path.insert(0, caffe_root)
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import tempfile

caffe.set_mode_cpu()


def _create_file_from_netspec(netspec):
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(netspec.to_proto()))
    return f.name


def get_complexity(netspec=None, prototxt_file=None, mode=None):
    # One of netspec, or prototxt_path params should not be None
    assert (netspec is not None) or (prototxt_file is not None)

    if netspec is not None:
        prototxt_file = _create_file_from_netspec(netspec)

    net = caffe.Net(prototxt_file, caffe.TEST)

    total_params = 0
    total_flops = 0

    net_params = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt_file).read(), net_params)
    print '\n ########### output ###########'
    for layer in net_params.layer:
        if layer.name in net.params:

            params = net.params[layer.name][0].data.size
            # If convolution layer, multiply flops with receptive field
            # i.e. #params * datawidth * dataheight
            if layer.type == 'Convolution':  # 'conv' in layer:
                data_width = net.blobs[layer.name].data.shape[2]
                data_height = net.blobs[layer.name].data.shape[3]
                flops = net.params[layer.name][
                    0].data.size * data_width * data_height
                # print >> sys.stderr, layer.name, params, flops
            else:
                flops = net.params[layer.name][0].data.size
            flops *= 2 
            print('%s: #params: %s, #FLOPs: %s') % (
                layer.name,
                digit2string(params),
                digit2string(flops))
            total_params += params
            total_flops += flops

    if netspec is not None:
        os.remove(prototxt_file)

    return total_params, total_flops


def digit2string(x):
    x = float(x)
    if x < 10 ** 3:
        return "%.2f" % float(x)
    elif x < 10 ** 6:
        x = x / 10 ** 3
        return "%.2f" % float(x) + 'K'
    elif x < 10 ** 9:
        x = x / 10 ** 6
        return "%.2f" % float(x) + 'M'
    else:
        x = x / 10 ** 9
        return "%.2f" % float(x) + 'B'


if __name__ == '__main__':
    length = len(sys.argv)
    if length == 1:
        filepath = 'deploy.prototxt'
    else:
        filepath = sys.argv[1]
    params, flops = get_complexity(prototxt_file=filepath, mode='Test')
    print '\n ########### result ###########'
    print '#params=%s, #FLOPs=%s' % (digit2string(params),
                                     digit2string(flops))
