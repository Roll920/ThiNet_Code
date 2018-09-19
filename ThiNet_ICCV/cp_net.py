import os
import shutil
import re
import sys

if __name__ == '__main__':
    compress_layer = int(sys.argv[1])
    layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1',
                 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
    FileNames = os.listdir(layers[compress_layer]+'/snapshot/')
    max_value = 0
    for i in range(len(FileNames)):
        name = re.findall(r"_iter_(.+?).caffemodel", FileNames[i])
        if len(name) > 0:
            if int(name[0]) > max_value:
                max_value = int(name[0])
    filename = layers[compress_layer]+'/snapshot/_iter_'+str(max_value)+".caffemodel"
    shutil.copyfile(filename, "model.caffemodel")
