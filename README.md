# Deep-Unsupervised-Pixelization
## Paper
[Deep Unsupervised Pixelization](http://www.cse.cuhk.edu.hk/~ttwong/papers/pixel/pixel.pdf) and [Supplementary Material](http://www.cse.cuhk.edu.hk/~ttwong/papers/pixel/pixel-supp.pdf).  
Chu Han^, Qiang Wen^, Shengfeng He*, Qianshu Zhu, Yinjie Tan, Guoqiang Han, and Tien-Tsin Wong. (^joint first authors)  
ACM Transactions on Graphics (SIGGRAPH Asia 2018 issue), 2018.  
## ![Our teaser](./teaser/teaser.png)
## Requirement
- Python 3.6.13
- PIL
- Numpy
- Pytorch 0.4.1
- Ubuntu 16.04 LTS
- pip install torch==0.4.1 -f https://download.pytorch.org/whl/torch
- torchvision 0.2.1

- Package           Version
----------------- ---------
certifi           2021.5.30
cffi              1.14.6
dataclasses       0.8
mkl-fft           1.0.6
mkl-random        1.0.1
numpy             1.15.4
Pillow            8.4.0
pip               21.2.2
pycparser         2.21
setuptools        58.0.4
six               1.16.0
TBB               0.2
torch             0.4.1
torchvision       0.2.1
typing_extensions 4.1.1
wheel             0.37.1

## Dataset
### Training Dataset
Create the folders `trainA` and `trainB` in the directory `./samples/`. Note that `trainA` and `trainB` contain the clip arts to be pixelized and pixel arts to be depixelized respectively.
### Testing Dataset
Create the folders `testA` and `testB` in the directory `./samples/`. Note that `testA` and `testB` contain the clip arts to be pixelized and pixel arts to be depixelized respectively.
## Training
* To train a model:
``` bash
python3 ./train.py --dataroot ./samples --resize_or_crop crop --gpu_ids 0
```  
or you can directly:
``` bash 
$ bash ./train.sh
```  
You can check the losses of models in the file `./checkpoints_pixelization/loss_log.txt`.  
More training flags in the files `./options/base_options.py` and `./options/train_options.py`.
## Testing
* After training, all models have been saved in the directory `./checkpoints_pixelization/`.
* To test a model:
``` bash
python3 ./test.py --dataroot ./samples --no_dropout --resize_or_crop crop --gpu_ids 0 --how_many 1 --which_epoch 200
```  
or you can directly:
``` bash 
$ bash ./test.sh
```  
More testing flags in the file `./options/base_options.py`.  
All testing results will be shown in the directory `./results_pixelization/`.
## Note
Since this proposed method has been used in commerce, we cannot release the pretrained model and training dataset.
## Acknowledgments
Part of the code is based upon [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
## Citation
```
@article{han2018deep,
  title={Deep unsupervised pixelization},
  author={Han, Chu and Wen, Qiang and He, Shengfeng and Zhu, Qianshu and Tan, Yinjie and Han, Guoqiang and Wong, Tien-Tsin},
  journal={ACM Transactions on Graphics (TOG)},
  volume={37},
  number={6},
  pages={1--11},
  year={2018},
  publisher={ACM New York, NY, USA}
}
```
