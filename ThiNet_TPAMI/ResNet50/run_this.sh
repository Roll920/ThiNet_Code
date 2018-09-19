#!/bin/bash
# cp /data/luojh/net/caffe/ResNet-50-model.caffemodel model.caffemodel
layers=(2a 2b 2c 3a 3b 3c 3d 4a 4b 4c 4d 4e 4f 5a 5b 5c)
TOOLS=/home/luojh2/Software/caffe-master/build/tools
gpu=4
gpus=4,5,6,7
compression_rate=0.7

for compress_layer in $(seq 4 15)
do
	python compress_model.py ${compress_layer} 0 ${compression_rate} ${gpu}
	python compress_model.py ${compress_layer} 1 ${compression_rate} ${gpu}

	compress_block=1
    log_name="ResNet_50.log"
    LOG=${layers[compress_layer]}_$compress_block/logs/${log_name}    
    if [ ! -d "${layers[compress_layer]}_$compress_block/logs" ]; then
        mkdir ${layers[compress_layer]}_$compress_block/logs
    fi

    solver_path=${layers[compress_layer]}_$compress_block/solver.prototxt
    $TOOLS/caffe train --solver=$solver_path -weights model.caffemodel -gpu ${gpus} 2>&1 | tee $LOG

    cd ${layers[compress_layer]}_$compress_block/logs/
    ../../parse_log/parse_log.sh "$log_name"
    cd ../..

    python cp_net.py ${compress_layer} ${compress_block}
done
