# [AAAI 2024] DDRNet
This is the official PyTorch implementation of the paper [Decoupling Degradations with Recurrent Network for Video Restoration in Under-Display Camera](https://).

## Contents
- [Introduction](#introduction)
  - [Contribution](#contribution)
  - [Overview](#overview)
  - [Visual](#Visual)
- [Dataset](#dataset)
- [Test](#test)
- [Train](#train)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)

## Introduction
<img src="./fig/teaser.png" width=100%>

### Contribution
* We propose a novel network with long- and short-term video representation learning by decoupling video degradations for the UDC video restoration task (D$^2$RNet), which is the first work to address UDC video degradation. The core decoupling attention module (DAM) enables a tailored solution to the degradation caused by different incident light intensities in the video. 
* We propose a large-scale UDC video restoration dataset (VidUDC33K), which includes numerous challenging scenarios. To the best of our knowledge, this is the first dataset for UDC video restoration.
* Extensive quantitative and qualitative evaluations demonstrate the superiority of D$^2$RNet. In the proposed VidUDC33K dataset, D$^2$RNet gains 1.02db PSNR improvements more than other restoration methods.
    
### Overview
<img src="./fig/overview.jpg" width=100%>

### Visual
<img src="./fig/result.jpg" width=90%>


## Model and Results
Pre-trained models can be downloaded from [onedrive](https://1drv.ms/u/s!Au4fJlmAZDhlhwjmP0D2RJOQaFqF?e=UHVz3H), [google drive](https://drive.google.com/drive/folders/1JWl22XUc0IOp1mx79_DRtwOwHjO1FP8I?usp=sharing), and [baidu cloud](https://pan.baidu.com/s/1nCjVhwArNajWFDDYwt4IUA)(j3nd).
* *TTVFI_stage1.pth*: trained from first stage with consistent motion learning.
* *TTVFI_stage2.pth*: trained from second stage with trajectory-aware Transformer on Viemo-90K dataset.

The output results on Vimeo-90K testing set, DAVIS, UCF101 and SNU-FILM can be downloaded from [onedrive](https://1drv.ms/u/s!Au4fJlmAZDhlhwjmP0D2RJOQaFqF?e=UHVz3H), [google drive](https://drive.google.com/drive/folders/1JWl22XUc0IOp1mx79_DRtwOwHjO1FP8I?usp=sharing), and [baidu cloud](https://pan.baidu.com/s/1nCjVhwArNajWFDDYwt4IUA)(j3nd).


## Dataset
1. Download the [Video](http://) under `./dataset`.
2. Generate the sequences by `xxx.txt` file listing the training samples in the download zip file.
```
cd ./models/PWCNet/correlation_package_pytorch1_0/
./build.sh
```
<img src="./fig/build_dataset.jpg" width=100%>
3. Prepare input frames and modify "FirstPath" and "SecondPath" in `./demo.py`
4. Make VidUDC33K structure be:
        ```
			├────VidUDC33K
				├────Input
					├────000
                                       	        ├────000.npy
                                        	├────...
                                        	├────049.npy
					├────001
					├────...
					├────676
				├────GT
					├────000
                                       	        ├────000.npy
                                        	├────...
                                        	├────049.npy
					├────001
					├────...
					├────676
        ```
<img src="./fig/dataset.jpg" width=80%>

## Test
1. Clone this github repo
```
git clone https://github.com/ChengxuLiu/DDRNet.git
cd DDRNet
```
2. Prepare testing dataset and modify "folder_lq" and "folder_lq" in `./test.py`
3. Run test
```
python test.py --save_result
```
4. The result are saved in `./results`

## Train
1. Clone this github repo
```
git clone https://github.com/ChengxuLiu/DDRNet.git
cd DDRNet
```
2. Prepare training dataset and modify "dataroot_gt" and "dataroot_lq" in `./options/DDRNet/train_DDRNet.json`
3. Run training
```
python train.py --opt ./options/DDRNet/train_DDRNet.json
```
4. The models are saved in `./experiments`


## Citation
If you find the code and pre-trained models useful for your research, please consider citing our paper. :blush:
```
@article{liu2023ttvfi,
  title={Ttvfi: Learning trajectory-aware transformer for video frame interpolation},
  author={Liu, Chengxu and Yang, Huan and Fu, Jianlong and Qian, Xueming},
  journal={IEEE Transactions on Image Processing},
  year={2023},
  publisher={IEEE}
}
``` 

## Contact
If you meet any problems, please describe them in issues or contact:
* Chengxu Liu: <liuchx97@gmail.com>

## Acknowledgement
The code of DDRNet is built upon [RVRT](https://github.com/JingyunLiang/RVRT) and [MMagic](https://github.com/open-mmlab/mmagic), and we express our gratitude to these awesome projects.


