#!/bin/bash
cp /data/luojh/net/caffe/VGG16/VGG_ILSVRC_16_layers.caffemodel model.caffemodel
layers=(conv1_1 conv1_2 conv2_1 conv2_2 conv3_1 conv3_2 conv3_3 conv4_1 conv4_2 conv4_3 conv5_1 conv5_2 conv5_3 fc6 fc7)
TOOLS=/home/luojh2/Software/caffe-master/build/tools
gpus=0,1,2,3
gpu=0
compression_rate=0.5

for compress_layer in $(seq 0 9)
do
    python compress_model.py ${compress_layer} ${gpu} ${compression_rate}
    
    log_name="value_sum_vgg.log"
    LOG=${layers[compress_layer]}/logs/${log_name}
    if [ ! -d "${layers[compress_layer]}/logs" ]; then
       mkdir ${layers[compress_layer]}/logs
    fi

    solver_path=${layers[compress_layer]}/solver.prototxt
    $TOOLS/caffe train --solver=$solver_path -weights model.caffemodel -gpu ${gpus} 2>&1 | tee $LOG

    cd ${layers[compress_layer]}/logs/
    ../../parse_log/parse_log.sh "$log_name"
    cd ../..

    python cp_net.py ${compress_layer}
done
