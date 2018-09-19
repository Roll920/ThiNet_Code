# Caffe Implementation of ThiNet
* ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression, ICCV 2017.
* ThiNet: Pruning CNN Filters for a Thinner Net, TPAMI, 2018.
* [[ICCV Project Page]](http://lamda.nju.edu.cn/luojh/project/ThiNet_ICCV17/ThiNet_ICCV17.html)   [[Pretrained Models]](https://github.com/Roll920/ThiNet)

## Requirements 
Python 2.6 & Caffe environment:
* Python2.6
* Caffe & Caffe's Python interface

## Usage
1. Clone the ThiNet repository.
2. select ThiNet_ICCV or ThiNet_TPAMI subfolder:
   ```
   cd ThiNet_ICCV
   ```
3. modify your configuration path:
   + modify the *caffe*  path (caffe_root) at the beginning of `net_generator.py` and `compress_model.py`
   + modify ImageNet *lmdb* file path in line 212 and line 217 of `net_generator.py`
   + modify ImageNet *dataset* path in line 54, 55, 60 of `compress_model.py`
   + modify line 2 and 4 in `run_this.sh` with correct file path.
4. Run the pruning demo:
   ```
   ./run_this.sh
   ```

## Other Toolkits
* Image Resize:
  1. Note that there are two different strategies to organize ImageNet dataset:
     + fixed size: each image is firstly resized to 256×256, then center-cropped to obtain a 224×224 regin;
     + keep aspect ratio: each image is firstly resized with shorter side=256, then center-cropped;
  2. The default caffe `create_lmdb.sh` file will convert images into 256x256. If you want to keep the original ratio: 
     + replace `caffe/src/caffe/util/io.cpp` with `toolkit/caffe_lmdb_keep_ratio/io.cpp` 
     + rebuild caffe
     + use the provided script `toolkit/caffe_lmdb_keep_ratio/create_lmdb.sh` to create the lmdb file. 
     + do not forget to modify the configuration path of this script.

* FLOPs Calculation:
  ```
  cd toolkit
  modify the caffe_root at the beginning of FLOPs_and_size.py file.
  python FLOPs_and_size.py [the path of *.prototxt file]
  ```
  **NOTE:** we regard the vector multiplication as **TWO** float-point operations (multiplication and addition). In some paper,  it is calculated as **ONE** operation. Do not be confused if the result is twice larger.

## Results
We prune the [VGG_ILSVRC_16_layers model](https://gist.github.com/ksimonyan/211839e770f7b538e2d8) on ImageNet dataset with ratio=0.5:

| Method  | Top-1 Acc.  | Top-5 Acc.  | #Param.   | #FLOPs  |
| ------------- | ------------- | ------------- |  ------------- |  ------------- | 
| original VGG16  | 71.50%  | 90.01%  | 138.24M  | 30.94B  |
|  ThiNet_ICCV |  69.80%  | 89.53%  | 131.44M  | 9.58B  |
| ThiNet_TPAMI | 69.74% | 89.41% | 131.44M | 9.58B |

There are no difference on [VGG16](https://gist.github.com/ksimonyan/211839e770f7b538e2d8), but ThiNet_TPAMI is much better on [ResNet50](https://github.com/KaimingHe/deep-residual-networks):

| Method  | Top-1 Acc.  | Top-5 Acc.  | #Param.   | #FLOPs  |
| ------------- | ------------- | ------------- |  ------------- |  ------------- | 
| original ResNet50  | 75.30%  | 92.20%  | 25.56M  | 7.72B  |
|  ThiNet_ICCV |  72.04%  | 90.67%  | 16.94M | 4.88B |
| ThiNet_TPAMI | 74.03% | 92.11% | 16.94M | 4.88B |

## Citation
If you find this work is useful for your research, please cite:
```
@CONFERENCE{ThiNet_ICCV17,
  author={Jian-Hao Luo, Jianxin Wu, and Weiyao Lin},
  title={ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression},
  booktitle={ICCV},
  year = {2017},
  pages={5058-5066},
}
```
```
@article{ThiNet_TPAMI,
  author = {Jian-Hao Luo, Hao Zhang, Hong-Yu Zhou, Chen-Wei Xie, Jianxin Wu, and Weiyao Lin},
  title = {ThiNet: Pruning CNN Filters for a Thinner Net},
  journal = {IEEE Trans. on Pattern Analysis and Machine Intelligence},
  year = 2008,
}
```

## Contact
Feel free to contact me if you have any question (Jian-Hao Luo luojh@lamda.nju.edu.cn or jianhao920@gmail.com).

