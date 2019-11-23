# Location-aware-Upsampling [[arXiv]](https://arxiv.org/abs/1911.05250)
Pytorch implementation of "Location-aware Upsampling for Semantic Segmentation" (LaU). If you are only interested in the upsampling part, please refer to [LaU.md](./LaU.md). 

### 1. Dependencies :
* **Python 3.5.6**
* **PyTorch 1.0.0**
* **GCC 7.3.0**

### 2. Usage :
##### 2.1 install requirements
```sh
conda create -n LaU python=3.5
conda activate LaU
pip install ninja cython numpy nose tqdm cffi==1.0.0 Pillow scipy requests torch==1.0.0 torchvision==0.2.2.post2 
cd ./LaU-reg
# install detail-api
cd ./detail-api/PythonAPI/
python setup.py install
# install LaU
cd ../../
python setup.py develop # python setup.py install
cp -r ../LaU ./encoding/models/
cd ./encoding/models/LaU
bash make.sh
```
##### 2.2 Prepare data
```sh
cd LaU-reg
# E.g., download pcontext
python scripts/prepare_pcontext.py
# python scripts/prepare_ade20k.py
```
##### 2.3 Train / Val / Test
```sh
cd LaU-reg/experiments/segmentation
# E.g., train and evaluate EncNet101-LaU4x-reg on pcontext
# train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --dataset pcontext \
    --model encnet --dilated --lateral --aux --se-loss --offset-loss --batch-size 16\
    --backbone resnet101 --checkname encnet_resnet101_pcontext --no-val --offset-weight 0.3 --location-weight 0.1 --up-factor 4 --batch-size-per-gpu 2 --bottleneck-channel 64 --offset-branch-input-channel 512 --category 150 --base-size 520 --crop-size 480 --downsampled-input-size 60
# val [single]
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --dataset pcontext \
    --model encnet --dilated --lateral --aux --se-loss --offset-loss \
    --backbone resnet101 --resume /home/shuan/LaU-reg/experiments/segmentation/runs/encnet_resnet101_pcontext/encnet/encnet_resnet101_pcontext/179_checkpoint.pth.tar --split val --mode test --up-factor 4 --batch-size-per-gpu 1 --bottleneck-channel 64 --offset-branch-input-channel 512 --category 150 --base-size 520 --crop-size 480 --downsampled-input-size 60 # --ms  
```
Or
```sh
cd LaU-reg/experiments/segmentation
# E.g., train and evaluate EncNet101-LaU4x-reg on pcontext
bash ./scripts/encnet_res101_pcontext.sh 2>&1 | tee ./log.txt
```
More scripts can be found in `LaU-reg/experiments/segmentation/scripts/`.

### 3. Main difference between LaU and EncNet
1. We replace the original `SegmentationLosses` with `OffsetLosses` (defined in `LaU-reg/encoding/nn/customize.y` and used in `LaU-reg/experiments/segmentation/train.py`). 
2. We add an LaU to the top of the decoder in EncNet (defined in `LaU-reg/encoding/models/encnet.py`).
3. The LaU module can be found in `LaU-reg/encoding/models/util.py` and `LaU-reg/encoding/models/lau.py`.

### 4. Acknowledgement :
We would like to thank [FastFCN](https://github.com/wuhuikai/FastFCN) and [Encoding](https://github.com/zhanghang1989/PyTorch-Encoding) for sharing their codes!

### 5. Cite : 
If you find this project helpful, please cite
```bib
@misc{he2019locationaware,
    title={Location-aware Upsampling for Semantic Segmentation},
    author={Xiangyu He and Zitao Mo and Qiang Chen and Anda Cheng and Peisong Wang and Jian Cheng},
    year={2019},
    eprint={1911.05250},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
