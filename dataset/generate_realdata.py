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


def convert_npy_as_img(np1):
    img = (np1 / (np1 + 0.25))
    img = cv2.cvtColor((img*255.0).astype(np.uint8),cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (1920,1080))
    img = img[28:1052,:,:]
    return img


def convert_npy_as_img_real(np1):
    img = cv2.cvtColor((np1*255.0).astype(np.uint8),cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (1080,1920))
    img = img[:,28:1052,:]
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', help='Specify path to dataset root directory eg. "./"', 
                        type=str, default='./Real_video/')
    
    parser.add_argument('--save_path', help='Specify path to dataset save directory eg. "./"', 
                        type=str, default='./VidUDC33K_real/')
    
    parser.add_argument('--frame_count', help='Frame count of every sequence.', 
                        type=int, default=50)
    
    parser.add_argument('--save_img', help='Whether or not to save 8 bit image.', 
                        type=bool, default=True)
    
    parser.add_argument('--save_img_path', help='Save 8 bit image filepath.', 
                        type=str, default='./VidUDC33K_real_png/')
    
    parser.add_argument('--txt_path', help='Build Dataset from txt file.', 
                        type=str, default='./realvideo_meta.txt')

    args = parser.parse_args()
    video_lists = glob.glob(os.path.join(args.video_path,'*.mp4'))

    dataset_path = args.save_path
    frame_count = args.frame_count
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    #mkdir /RealVideo/Input
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
                cut_list.append([string_origin[1],stroffset2frame(string_origin[3],string_origin[5],fps=24)])
        reader = imageio.get_reader(video, format='ffmpeg', dtype='uint16')
        
        current_name = None
        start_frame = None
        pre_frame = None
        H = None
        H_psf = None
        cnt = 0
        flag = 0
        for i, current_frame in enumerate(reader):
            for j in range(0,len(cut_list)):
                if i == cut_list[j][1] and flag == 0:
                    flag = 1
                    start_frame = cut_list[j][1]
                    current_name = cut_list[j][0]
                    print(start_frame,current_name)
                    print("{}---{},Start!".format(video_name,current_name))
                    
                elif i == cut_list[j][1] and flag == 1:
                    cnt += 1
                    
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
                print(i,start_frame)

                img_ori = current_frame.astype(np.float32) / 65535

                first_filepath = str(current_name).zfill(3)
                print(first_filepath)

                second_filepath = str(int((i - start_frame) / 3)).zfill(3) + '.npy'
                print(second_filepath)

                first_input_filepath = os.path.join(input_path,first_filepath)
                if not os.path.exists(first_input_filepath):
                    os.makedirs(first_input_filepath)
                second_input_filepath = os.path.join(first_input_filepath,second_filepath)
                np.save(second_input_filepath, img_ori.astype(np.float16) )

                if args.save_img:
                    if not os.path.exists(args.save_img_path):
                        os.makedirs(args.save_img_path)
                    second_img_filepath = str(int((i - start_frame) / 3)).zfill(3) + '.png'
                    
                    save_ori_path = os.path.join(args.save_img_path,first_filepath)
                    if not os.path.exists(save_ori_path):
                        os.makedirs(save_ori_path)
                    save_ori_path = os.path.join(save_ori_path,second_img_filepath)
                    cv2.imwrite(save_ori_path,convert_npy_as_img_real(img_ori))




