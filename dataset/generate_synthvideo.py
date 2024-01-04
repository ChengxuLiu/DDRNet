import os
import sys
import cv2
import glob
import numpy as np
import imageio
import argparse
import tqdm
import torch


def str2sec(x):
    h, m, s = x.strip().split(':')
    return int(h)*3600 + int(m)*60 + int(s)


def stroffset2frame(time,offset,fps):
    seconds = str2sec(time)
    return seconds * fps + int(offset)


def pad_edges(data, dim):
    pad_h, pad_w = [max(dim[0] - data.shape[0], 0), 
                    max(dim[1] - data.shape[1], 0)]
    pad_top = pad_bot = pad_h // 2
    pad_left = pad_right = pad_w // 2
    
    if pad_h % 2 != 0:
        pad_bot += 1
    if pad_w % 2 != 0:
        pad_right += 1
    pad_tuple = ((pad_top, pad_bot), (pad_left, pad_right))
    if len(data.shape) == 3:
        pad_tuple = pad_tuple + ((0, 0),)
    return np.pad(data, pad_width=pad_tuple,mode='constant')


def center_crop(data, dim):
    h_start, w_start = [max(data.shape[0] - dim[0], 0) // 2,
                        max(data.shape[1] - dim[1], 0) // 2]
    h_end, w_end = [h_start + min(dim[0], data.shape[0]),
                    w_start + min(dim[1], data.shape[1])]
    return data[h_start:h_end, w_start:w_end]


def match_dim(data, dim):
    if data.shape[0] < dim[0] or data.shape[1] < dim[1]:
        data = pad_edges(data, dim[:2])  
    if data.shape[0] > dim[0] or data.shape[1] > dim[1]:
        data = center_crop(data, dim[:2])
    return data


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def torch_rfft2(data):
    """
    Apply centered 2-dimensional Real-to-Complex Fast Fourier Transform.

    Args:
        data (torch.Tensor): Real valued input data containing at least 2 dimensions: dimensions
            -2 & -1 are spatial dimensions. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The FFT of the input where dimensions -3 & -2 are now spatial dimensions 
        and dimension -1 has size 2 for real & complex values.
    """
    assert data.size(-1) != 1
    data = ifftshift(data, dim=(-2, -1))
    data = torch.fft.fft2(data,dim=(-2,-1))
    data = torch.stack((data.real, data.imag), -1)    
    data = fftshift(data, dim=(-3, -2))
    return data


def torch_irfft2(data):
    """
    Apply centered 2-dimensional Complex-to-Real Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The IFFT of the input where dimensions -2 & -1 are now spatial dimensions.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3,-2))
    data = torch.fft.ifft2(torch.complex(data[..., 0],data[..., 1]), dim=(-2, -1))
    data = fftshift(data, dim=(-2,-1))
    return data


def TorchComplexMul( v1_complex, v2_complex ):
    ( v1_real, v1_imag ) = v1_complex.chunk(2, dim=-1)
    ( v2_real, v2_imag ) = v2_complex.chunk(2, dim=-1)
    return torch.cat( ( (v1_real * v2_real) - (v1_imag * v2_imag), (v1_real * v2_imag) + (v1_imag * v2_real)  ), dim = -1 )


def TorchFFTConv2d(a, K):
    """
    FFT tensor convolution of image a with kernel K 
    
    Args:
        a (torch.Tensor):   1-channel Image as tensor with at least 2 dimensions. 
                            Dimensions -2 & -1 are spatial dimensions and all other
                            dimensions are assumed to be batch dimensions
        K (torch.Tensor):   1-channel kernel as tensor with at least 2 dimensions.
    Return:
        Absolute value of the convolution of image a with kernel K 
    """
    K = torch_rfft2(K)
    a = torch_rfft2(a)

    img_conv = TorchComplexMul(K, a)
    img_conv = torch_irfft2(img_conv)
    
    return (img_conv**2).sqrt().cpu()


def psf_conv_torch(img,psf,pad_size):

    img = img.astype(np.float32) / 65535.0
    img_ori = img.copy()
    img_test_mean = np.mean(img, axis=2)
    img_test_max = np.max(img, axis=2)
    # print(img_test_mean.tolist())
    hight_light_mask = (img_test_mean > img_test_mean.max()*0.7) * (img_test_max > 0.85) 

    hight_light_mask = hight_light_mask.astype('uint8')[:,:,np.newaxis]
    contours = cv2.findContours(hight_light_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cnts = contours[0]
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h<36:
            hight_light_mask[y:y+h,x:x+w,:] = hight_light_mask[y:y+h,x:x+w,:]*100
        elif w*h<64:
            hight_light_mask[y:y+h,x:x+w,:] = hight_light_mask[y:y+h,x:x+w,:]*64
        else:
            hight_light_mask[y:y+h,x:x+w,:] = hight_light_mask[y:y+h,x:x+w,:]*8
    hight_light_mask = hight_light_mask+1

    img_gt = img * cv2.merge([hight_light_mask, hight_light_mask, hight_light_mask])
    h, w, _ = img_gt.shape  
    pad_img = pad_edges(img_gt, (h + pad_size*2, w + pad_size*2))
    psf_matched = psf
    if psf_matched.shape[0] != pad_img.shape[0] or psf_matched.shape[1] != pad_img.shape[1]:
        psf_matched = match_dim(psf_matched, pad_img.shape[:2])
    img_sim = np.zeros_like(img_gt)
    for c in range(3):
        img_sim[..., c] = center_crop(TorchFFTConv2d(torch.tensor(pad_img[..., c]).to('cuda:0'),torch.tensor(psf_matched[..., c]).to('cuda:0')).numpy(), (h, w))
        img_sim = np.clip(img_sim, a_min=0, a_max=500)
    return img_sim,img_gt,img_ori


def sift_kp(image):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp,des = sift.detectAndCompute(image,None)
    kp_image = cv2.drawKeypoints(gray_image,kp,None)
    return kp_image,kp,des


def get_good_match(des1,des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    return good


def cal_homography_martix(image1,image2):
    image1 = cv2.normalize(image1,None,0,255,cv2.NORM_MINMAX).astype('uint8')
    image2 = cv2.normalize(image2,None,0,255,cv2.NORM_MINMAX).astype('uint8')
    _,kp1,des1 = sift_kp(image1)
    _,kp2,des2 = sift_kp(image2)
    try:
        good_matches = get_good_match(des1,des2)
    except:
        good_matches = []

    MIN_NUM_GOOD_MATCHES = 10
    H = None
    if len(good_matches) >= MIN_NUM_GOOD_MATCHES:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return H


def convert_npy_as_img(np1):
    img = (np1 / (np1 + 0.25))
    img = cv2.cvtColor((img*255.0).astype(np.uint8),cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (1920,1080))
    # img = img[28:1052,:,:]
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', help='Specify path to dataset root directory eg. "./"', 
                        type=str, default='./Video/')
        
    parser.add_argument('--psf_path', help='PSF from screen version',
                        type=str, default='./ZTE_new_psf_5.npy')

    parser.add_argument('--pad_size', help='Padding size set to avoid boundary effects caused by FFT', 
                        type=int, default=200)
    
    parser.add_argument('--save_path', help='Specify path to dataset save directory eg. "./"', 
                        type=str, default='./VidUDC33K/')
    
    parser.add_argument('--frame_count', help='Frame count of every sequence.', 
                        type=int, default=50)
    
    parser.add_argument('--save_img', help='Whether or not to save 8 bit image.', 
                        type=bool, default=True)
    
    parser.add_argument('--save_img_path', help='Save 8 bit image filepath.', 
                        type=str, default='./VidUDC33K_png/')
    
    parser.add_argument('--txt_path', help='Build Dataset from txt file.', 
                        type=str, default='./synthvideo_meta.txt')

    args = parser.parse_args()
    psf = np.load(args.psf_path)
    video_lists = glob.glob(os.path.join(args.video_path,'*.mp4'))
    video_lists = ['./Video/TokyoChristmasLights.mp4', './Video/TokyoTowerWTC.mp4', './Video/YokohamaNightWalk.mp4', './Video/VeniceCarnival.mp4', './Video/TokyoLightTrail.mp4', './Video/Alaska.mp4', 
                   './Video/AmazingBeauty1080p.mp4', './Video/Best2022.mp4', './Video/Bulgaria.mp4', './Video/COSTA_RICA.mp4', './Video/DUBAI.mp4', './Video/Future1080p.mp4',
                   './Video/Liquid.mp4', './Video/Maldives.mp4', './Video/MorningSunrise.mp4', './Video/Morocco1080p.mp4', './Video/Peru.mp4', './Video/The_World.mp4']
    
    dataset_path = args.save_path
    frame_count = args.frame_count
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    #makedir /VideoUDC/Train
    gt_path = os.path.join(dataset_path,'GT',)
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)
    
    #makedir /VideoUDC/Input
    input_path = os.path.join(dataset_path,'Input')
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    tmp_first = 0
    for video in video_lists:
        video_name = os.path.split(video)[-1]
        save_txt = open(args.txt_path,encoding='utf-8')
        cut_list = list()
        for index, line in enumerate(save_txt.readlines()):
            string_origin = line.strip().split("'")
            if string_origin[7] == video_name:
                cut_list.append([string_origin[1],stroffset2frame(string_origin[3],string_origin[5],fps=60)])
        reader = imageio.get_reader(video, format='ffmpeg', dtype='uint16')
        
        
        current_name = None
        start_frame = None
        pre_frame = None
        H = None
        H_psf = None
        cnt = 0
        flag = 0
        for i, current_frame in enumerate(reader):
            print("video:",video, "   frame:", i )
            
            for j in range(0,len(cut_list)):
                if i == cut_list[j][1] and flag == 0:
                    flag = 1
                    start_frame = cut_list[j][1]
                    current_name = cut_list[j][0]
                    print("{}---{},Start!".format(video_name,current_name))
                    
            if cnt >= len(cut_list):
                print('A video is finished!')
                break
            if start_frame is None or current_name is None:
                continue
            
            if int(i - start_frame) >= 150 :
                cnt += 1
                current_name = None
                start_frame = None
                pre_frame = None
                H = None
                H_psf = None
                flag = 0

            if current_name is not None and (i - start_frame)%3==0:

                if pre_frame is None or (i - start_frame) % frame_count==0:
                    current_psf = psf
                else:
                    H = cal_homography_martix(pre_frame,current_frame)

                    if pre_psf is not None and H is not None:
                        current_psf = cv2.warpPerspective(pre_psf,H,(pre_psf.shape[1],pre_psf.shape[0]))
                    else:
                        print("{} have no H frame".format(video))
                        current_psf = psf

                    a = np.where(current_psf==np.max(current_psf))
                    current_psf = np.roll(current_psf,(round(400 - a[0][0]),round(400 - a[1][0])),(0,1))

                    

                img_sim,img_gt,img_ori = psf_conv_torch(current_frame,current_psf,pad_size=args.pad_size)


                pre_frame = current_frame.copy()
                pre_psf = current_psf.copy()


                first_filepath = str(current_name).zfill(3)
                second_filepath = str(int((i - start_frame) / 3)).zfill(3) + '.npy'

                first_gt_filepath = os.path.join(gt_path,first_filepath)
                if not os.path.exists(first_gt_filepath):
                    os.makedirs(first_gt_filepath)
                second_gt_filepath = os.path.join(first_gt_filepath,second_filepath)
                print(second_gt_filepath)
                np.save(second_gt_filepath,cv2.resize(img_gt,(1920,1080),interpolation=cv2.INTER_AREA).astype(np.float16))

                first_input_filepath = os.path.join(input_path,first_filepath)
                if not os.path.exists(first_input_filepath):
                    os.makedirs(first_input_filepath)
                second_input_filepath = os.path.join(first_input_filepath,second_filepath)
                np.save(second_input_filepath,cv2.resize(img_sim,(1920,1080),interpolation=cv2.INTER_AREA).astype(np.float16))

                if args.save_img and int((i - start_frame) / 3)%10==0:
                    if not os.path.exists(args.save_img_path):
                        os.makedirs(args.save_img_path)
                    second_img_filepath = str(int((i - start_frame) / 3)).zfill(3) + '.png'
                    save_gt_path = os.path.join(args.save_img_path,'GT',first_filepath)
                    if not os.path.exists(save_gt_path):
                        os.makedirs(save_gt_path)
                    save_gt_path = os.path.join(save_gt_path,second_img_filepath)
                    cv2.imwrite(save_gt_path,convert_npy_as_img(img_gt))

                    save_input_path = os.path.join(args.save_img_path,'Input',first_filepath)
                    if not os.path.exists(save_input_path):
                        os.makedirs(save_input_path)
                    save_input_path = os.path.join(save_input_path,second_img_filepath)
                    cv2.imwrite(save_input_path,convert_npy_as_img(img_sim))




