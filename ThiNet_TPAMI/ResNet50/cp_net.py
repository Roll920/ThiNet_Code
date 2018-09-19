import os
import shutil
import re
import sys

if __name__ == '__main__':
    compress_layer = int(sys.argv[1])
    compress_block = int(sys.argv[2])
    layers = ['2a', '2b', '2c', '3a', '3b', '3c', '3d', '4a', '4b', '4c', '4d', '4e', '4f', '5a', '5b', '5c']
    FileNames = os.listdir(layers[compress_layer]+'_'+str(compress_block)+'/snapshot/')
    max_value = 0
    for i in range(len(FileNames)):
        name = re.findall(r"_iter_(.+?).caffemodel", FileNames[i])
        if len(name) > 0:
            if int(name[0]) > max_value:
                max_value = int(name[0])
    filename = layers[compress_layer]+'_'+str(compress_block)+'/snapshot/_iter_'+str(max_value)+".caffemodel"
    shutil.copyfile(filename, "model.caffemodel")
