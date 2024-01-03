# [AAAI 2024] D$^2$RNet
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
1. Training set
	* [Viemo-90K](https://github.com/anchen1011/toflow) dataset. Download the [both triplet training and test set](http://). The `tri_trainlist.txt` file listing the training samples in the download zip file.
       - Make VidUDC33K structure be:
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
					├────001
					├────...
					├────676
        ```
2. Testing set
    * [Viemo-90K](https://github.com/anchen1011/toflow) testset. The `tri_testlist.txt` file listing the testing samples in the download zip file.
    * [DAVIS](https://github.com/HyeongminLEE/AdaCoF-pytorch/tree/master/test_input/davis), [UCF101](https://drive.google.com/file/d/0B7EVK8r0v71pdHBNdXB6TE1wSTQ/view?resourcekey=0-r6ihCy20h3kbgZ3ZdimPiA), and [SNU-FILM](https://myungsub.github.io/CAIN/) dataset.
		- Make DAVIS, UCF101, and SNU-FILM structure be:
		```
			├────DAVIS
				├────input
				├────gt
			├────UCF101
				├────1
				├────...
			├────SNU-FILM
				├────test
					├────GOPRO_test
					├────YouTube_test
				├────test-easy.txt			
				├────...		
				├────test-extreme.txt		
        ```
## Demo
1. Clone this github repo
```
git clone https://github.com/ChengxuLiu/TTVFI.git
cd TTVFI
```
2. Generate the Correlation package required by [PWCNet](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch/external_packages/correlation-pytorch-master):
```
cd ./models/PWCNet/correlation_package_pytorch1_0/
./build.sh
```
3. Download pre-trained weights ([onedrive](https://1drv.ms/u/s!Au4fJlmAZDhlhwjmP0D2RJOQaFqF?e=UHVz3H)|[google drive](https://drive.google.com/drive/folders/1JWl22XUc0IOp1mx79_DRtwOwHjO1FP8I?usp=sharing)|[baidu cloud](https://pan.baidu.com/s/1nCjVhwArNajWFDDYwt4IUA)(j3nd)) under `./checkpoint`
```
cd ../../..
mkdir checkpoint
```
4. Prepare input frames and modify "FirstPath" and "SecondPath" in `./demo.py`
5. Run demo
```
python demo.py
```



## Test
1. Clone this github repo
```
git clone https://github.com/ChengxuLiu/TTVFI.git
cd TTVFI
```
2. Generate the Correlation package required by [PWCNet](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch/external_packages/correlation-pytorch-master):
```
cd ./models/PWCNet/correlation_package_pytorch1_0/
./build.sh
```
3. Download pre-trained weights ([onedrive](https://1drv.ms/u/s!Au4fJlmAZDhlhwjmP0D2RJOQaFqF?e=UHVz3H)|[google drive](https://drive.google.com/drive/folders/1JWl22XUc0IOp1mx79_DRtwOwHjO1FP8I?usp=sharing)|[baidu cloud](https://pan.baidu.com/s/1nCjVhwArNajWFDDYwt4IUA)(j3nd)) under `./checkpoint`
```
cd ../../..
mkdir checkpoint
```
4. Prepare testing dataset and modify "datasetPath" in `./test.py`
5. Run test
```
mkdir weights
# Vimeo
python test.py
```

## Train
1. Clone this github repo
```
git clone https://github.com/ChengxuLiu/TTVFI.git
cd TTVFI
```
2. Generate the Correlation package required by [PWCNet](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch/external_packages/correlation-pytorch-master):
```
cd ./models/PWCNet/correlation_package_pytorch1_0/
./build.sh
```
3. Prepare training dataset and modify "datasetPath" in `./train_stage1.py` and `./train_stage2.py`
4. Run training of stage1
```
mkdir weights
# stage one
python train_stage1.py
```
5. The models of stage1 are saved in `./weights` and fed into stage2 (modify "pretrained" in `./train_stage2.py`)
6. Run training of stage2
```
# stage two
python train_stage2.py
```
7. The models of stage2 are also saved in `./weights`


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
The code of FSI is built upon [BNUDC](https://github.com/JaihyunKoh/BNUDC/tree/main) and [LaMa](https://github.com/advimman/lama), and we express our gratitude to these awesome projects.


