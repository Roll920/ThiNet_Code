# coding=utf-8
import sys
caffe_root = '/home/luojh2/Software/caffe-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe
from net_generator import solver_and_prototxt
import numpy as np
from PIL import Image
import random
import time
import os


def im_resize(im, height=224, width=224):
    d_type = im.dtype
    im = Image.fromarray(im)
    im = im.resize([height, width], Image.BICUBIC)
    im = np.array(im, d_type)
    return im


def convert2rgb(im):
    if len(im.shape) == 2:
        im = im.reshape((im.shape[0], im.shape[1], 1))
        im = np.concatenate((im, im, im), axis=2)
    if im.shape[2] == 4:
        im = np.array(Image.fromarray(im).convert('RGB'))
    return im


def get_index(compress_rate, compress_layer, gpu):
    # set parameters
    candidate = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1',
                 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
    relu_candidate = ['relu1_1', 'pool1', 'relu2_1', 'pool2', 'relu3_1', 'relu3_2', 'pool3', 'relu4_1',
                      'relu4_2', 'pool4', 'relu5_1', 'relu5_2', 'pool5', 'relu6', 'relu7']

    # load net
    model_def = candidate[compress_layer] + '/deploy.prototxt'
    model_weights = 'model.caffemodel'
    caffe.set_device(gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    mean_value = np.array([104, 117, 123], dtype=np.float32)
    mean_value = mean_value.reshape([3, 1, 1])

    # extract feature
    count = 0
    sample_num = 10  # for each category, we sample 10 images
    channel_num = 10  # for each image, we sample 10 locations
    padding = 1
    kernel_size = 3 # for vgg-16, the padding and kernel_size are same in each layer
    for foldername in os.listdir(r'/opt/luojh/Dataset/ImageNet/images/train'):
        img_list = os.listdir(r'/opt/luojh/Dataset/ImageNet/images/train/' + foldername)
        img_index = random.sample(range(len(img_list)), sample_num)
        for file_index in img_index:
            time_start = time.time()

            file_path = '/opt/luojh/Dataset/ImageNet/images/train/' + foldername + '/' + img_list[file_index]
            im = Image.open(file_path)
            im = convert2rgb(np.array(im))
            im = im_resize(im, 256, 256)
            im = np.array(im, np.float64)
            im = im[:, :, ::-1]  # convert RGB to BGR
            im = im.transpose((2, 0, 1))  # convert to 3x256x256
            im -= mean_value

            # shape for input (data blob is N x C x H x W), set data
            # center crop
            im = im[:, 16:240, 16:240]
            net.blobs['data'].reshape(1, *im.shape)
            net.blobs['data'].data[...] = im
            # run net and take argmax for prediction
            net.forward()

            Activation = net.blobs[candidate[compress_layer + 1]].data
            Input = net.blobs[relu_candidate[compress_layer]].data
            Filters = net.params[candidate[compress_layer + 1]][0].data
            Bias = net.params[candidate[compress_layer + 1]][1].data

            if compress_layer < 12:
                # conv1_1 to conv 5_2
                if count == 0:
                    X = np.zeros(
                        [channel_num * 1000 * sample_num, Filters.shape[1]])
                    Y = np.zeros([channel_num * 1000 * sample_num, 1])

                padded = np.zeros((Input.shape[0], Input.shape[1], Input.shape[
                                  2] + 2 * padding, Input.shape[3] + 2 * padding), dtype=np.float32)
                padded[:, :, padding:-padding, padding:-padding] = Input
                Input = padded

                for tmp in range(channel_num):
                    filter_num = random.randint(0, Filters.shape[0] - 1)
                    i = random.randint(0, Input.shape[2] - kernel_size)
                    j = random.randint(0, Input.shape[3] - kernel_size)
                    In_ = Input[:, :, i:i + kernel_size, j:j + kernel_size]
                    In_ = In_.reshape([In_.shape[1], -1])
                    F_ = Filters[filter_num, :, :, :]
                    F_ = F_.reshape([F_.shape[0], -1])
                    Out_ = Activation[0, filter_num, i, j] - Bias[filter_num]
                    X[count, :] = np.reshape(np.sum(F_ * In_, axis=1), [1, -1])
                    Y[count, 0] = np.reshape(Out_, [1, -1])
                    count += 1
            elif compress_layer == 12:
                # conv 5_3
                if count == 0:
                    X = np.zeros(
                        [channel_num * 1000 * sample_num, 512])
                    Y = np.zeros([channel_num * 1000 * sample_num, 1])

                for tmp in range(channel_num):
                    filter_num = random.randint(0, Filters.shape[0] - 1)
                    In_ = Input.reshape([Input.shape[1], -1])
                    F_ = Filters[filter_num, :]
                    F_ = F_.reshape([512, -1])
                    Out_ = Activation[:, filter_num] - Bias[filter_num]
                    X[count, :] = np.reshape(np.sum(F_ * In_, axis=1), [1, -1])
                    Y[count, 0] = np.reshape(Out_, [1, -1])
                    count += 1
            else:
                # fc6 fc7
                if count == 0:
                    X = np.zeros(
                        [channel_num * 1000 * sample_num, Filters.shape[1]])
                    Y = np.zeros([channel_num * 1000 * sample_num, 1])

                for tmp in range(channel_num):
                    filter_num = random.randint(0, Filters.shape[0] - 1)
                    In_ = Input.reshape(-1)
                    F_ = Filters[filter_num, :]
                    Out_ = Activation[:, filter_num] - Bias[filter_num]
                    X[count, :] = np.reshape(F_ * In_, [1, -1])
                    Y[count, 0] = np.reshape(Out_, [1, -1])
                    count += 1

            time_end = time.time()
            print 'Done! use %f second, %d image' % (time_end - time_start, count / channel_num)

    # select filters
    return value_sum(X, Y, compress_rate)


# use greedy method to select filters
# x:N*64 matrix, N is the instance number, 64 is channel number
def value_sum(x, y, compress_rate):
    # 1. set parameters
    time_s = time.time()
    x = np.mat(x)
    y = np.mat(y)
    goal_num = int(x.shape[1] * compress_rate)
    discard_num = int(x.shape[1]) - goal_num
    discard_index = []

    # 2. select
    x_tmp = 0
    for i in range(discard_num):
        min_value = float("inf")
        for j in range(x.shape[1]):
            if j not in discard_index:
                tmp_value = np.linalg.norm(x_tmp + x[:, j])
                if tmp_value < min_value:
                    min_value = tmp_value
                    min_index = j
        discard_index.append(min_index)
        x_tmp += x[:, min_index]
        print('discard num={0}, channel num={1}, i={2}, loss={3:.3f}'.format(discard_num, x.shape[1], i, min_value))

    # 3. return index
    index = set(range(x.shape[1])) - set(discard_index)
    index = np.array(list(index))
    index = np.sort(index)

    # 4.least square
    selected_x = x[:, index]
    w = (selected_x.T * selected_x).I * (selected_x.T * y)
    w = np.array(w)

    loss = np.linalg.norm(y - selected_x*w)
    print('loss before w={0:.3f}, loss with w={1:.3f}'.format(np.linalg.norm(y - np.sum(selected_x, 1)), loss))
    print 'Time used: %f s' % (time.time()-time_s)
    return index, w


def compress_net(index, w, compress_layer):
    # other layers
    candidate = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1',
                 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']

    model_def = candidate[compress_layer] + '/deploy.prototxt'
    model_weights = 'model.caffemodel'
    net = caffe.Net(model_def,  # defines the structure of the matrix
                    model_weights,  # contains the trained weights
                    caffe.TEST)  # use test mode (e.g., don't perform dropout)

    model_def = candidate[compress_layer] + '/trainval.prototxt'
    # use test mode (e.g., don't perform dropout)
    net_new = caffe.Net(model_def, caffe.TEST)

    current_layer = candidate[compress_layer]
    next_layer = candidate[compress_layer + 1]
    for i in range(0, len(candidate)):
        if candidate[i] != current_layer and candidate[i] != next_layer:
            net_new.params[candidate[i]][
                0].data[...] = net.params[candidate[i]][0].data
            net_new.params[candidate[i]][
                1].data[...] = net.params[candidate[i]][1].data

    weight = net.params[current_layer][0].data
    bias = net.params[current_layer][1].data
    if len(weight.shape) == 4:
        # conv layer
        weight = weight[index, :, :, :]
    else:
        # fc layer
        weight = weight[index, :]

    bias = bias[index]
    net_new.params[current_layer][0].data[...] = weight
    net_new.params[current_layer][1].data[...] = bias
    # next
    weight = net.params[next_layer][0].data
    bias = net.params[next_layer][1].data
    if len(weight.shape) == 4:
        # conv layer
        weight = weight[:, index, :, :]
        for i in range(weight.shape[1]):
            weight[:, i, :, :] *= w[i]
    elif next_layer == 'fc6':
        # fc6
        weight = weight.reshape([4096, 512, 7, 7])
        weight = weight[:, index, :, :]
        for i in range(weight.shape[1]):
            weight[:, i, :, :] *= w[i]
        weight = weight.reshape([4096, -1])
    else:
        # fc7 or fc8 layer
        weight = weight[:, index]
        for i in range(weight.shape[1]):
            weight[:, i] *= w[i]

    net_new.params[next_layer][0].data[...] = weight
    net_new.params[next_layer][1].data[...] = bias

    net_new.save('model.caffemodel')
    print 'OK!'


if __name__ == '__main__':
    compress_layer = int(sys.argv[1])
    gpu = int(sys.argv[2])
    compress_rate = float(sys.argv[3])

    solver_and_prototxt(compress_layer, compress_rate)
    index, w = get_index(compress_rate, compress_layer, gpu)

    compress_net(index, w, compress_layer)
