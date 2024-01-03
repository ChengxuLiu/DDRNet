# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import cv2
import glob
import os
import torch
import requests
import time
import numpy as np
from os import path as osp
from collections import OrderedDict
from torch.utils.data import DataLoader
from models.network_DDRNet import DDRNet as net


from utils import utils_image as util
from data.dataset_video_test import VideoRecurrentTestDataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='train_DDRNet', help='tasks')
    parser.add_argument('--folder_lq', type=str, default='/dataset/VidUDC33K/',
                        help='input low-quality test video folder')
    parser.add_argument('--folder_gt', type=str, default='/dataset/VidUDC33K/',
                        help='input ground-truth test video folder')
    parser.add_argument('--tile', type=int, nargs='+', default=[50,256,256],
    # parser.add_argument('--tile', type=int, nargs='+', default=[20,1024,1920],
                        help='Tile size, [0,0,0] for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, nargs='+', default=[2,20,20],
                        help='Overlapping of different tiles')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers in data loading')
    parser.add_argument('--save_result', action='store_true', help='save resulting image')
    args = parser.parse_args()

    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = prepare_model_dataset(args)
    model.eval()
    model = model.to(device)
    test_set = VideoRecurrentTestDataset({'dataroot_gt':args.folder_gt, 'dataroot_lq':args.folder_lq,
                                        'num_frame':-1, 'cache_data': False})
    test_loader = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=1, shuffle=False)

    save_dir = f'results/{args.task}'
    if args.save_result:
        os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    assert len(test_loader) != 0, f'No dataset found at {args.folder_lq}'

    for idx, batch in enumerate(test_loader):
        lq = batch['L'].to(device)
        folder = batch['folder']
        gt = batch['H'] if 'H' in batch else None

        # inference
        with torch.no_grad():
            print(lq.size())
            start = time.time()
            output = test_video(lq, model, args)
            end = time.time()
            print(end-start)
            print(output.size())


        test_results_folder = OrderedDict()
        test_results_folder['psnr'] = []
        test_results_folder['ssim'] = []
        test_results_folder['psnr_y'] = []
        test_results_folder['ssim_y'] = []

        for i in range(output.shape[1]):
            # save image
            img = output[:, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if img.ndim == 3:
                img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
            img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
            if args.save_result:
                seq_ = osp.basename(batch['lq_path'][i][0]).split('.')[0]
                os.makedirs(f'{save_dir}/{folder[0]}', exist_ok=True)
                cv2.imwrite(f'{save_dir}/{folder[0]}/{seq_}.png', img)

            # evaluate psnr/ssim
            if gt is not None:
                img_gt = gt[:, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if img_gt.ndim == 3:
                    img_gt = np.transpose(img_gt[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
                img_gt = np.squeeze(img_gt)

                test_results_folder['psnr'].append(util.calculate_psnr(img, img_gt, border=0))
                test_results_folder['ssim'].append(util.calculate_ssim(img, img_gt, border=0))
                if img_gt.ndim == 3:  # RGB image
                    img = util.bgr2ycbcr(img.astype(np.float32) / 255.) * 255.
                    img_gt = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                    test_results_folder['psnr_y'].append(util.calculate_psnr(img, img_gt, border=0))
                    test_results_folder['ssim_y'].append(util.calculate_ssim(img, img_gt, border=0))
                else:
                    test_results_folder['psnr_y'] = test_results_folder['psnr']
                    test_results_folder['ssim_y'] = test_results_folder['ssim']

        if gt is not None:
            psnr = sum(test_results_folder['psnr']) / len(test_results_folder['psnr'])
            ssim = sum(test_results_folder['ssim']) / len(test_results_folder['ssim'])
            psnr_y = sum(test_results_folder['psnr_y']) / len(test_results_folder['psnr_y'])
            ssim_y = sum(test_results_folder['ssim_y']) / len(test_results_folder['ssim_y'])
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            test_results['psnr_y'].append(psnr_y)
            test_results['ssim_y'].append(ssim_y)
            print('Testing {:20s} ({:2d}/{}) - PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                      format(folder[0], idx, len(test_loader), psnr, ssim, psnr_y, ssim_y))
        else:
            print('Testing {:20s}  ({:2d}/{})'.format(folder[0], idx, len(test_loader)))

    # summarize psnr/ssim
    if gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
        ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
        print('\n{} \n-- Average PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
              format(save_dir, ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y))


def prepare_model_dataset(args):
    ''' prepare model and dataset according to args.task. '''
    # define model
    if args.task == 'train_DDRNet':
        model = net(mid_channels=48,
                    scale_feat=12,
                    num_blocks=5,
                    max_residue_magnitude=10,
                    tau=0.9)
        datasets = ['UDCVideo']
        args.scale = 1
        args.window_size = [6,8,8]
        args.nonblind_denoising = False


    model_path = './checkpoints/DDRNet.pth'
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model['params'] if 'params' in pretrained_model.keys() else pretrained_model, strict=True)

    return model


def test_video(lq, model, args):
        '''test the video as a whole or as clips (divided temporally). '''

        num_frame_testing = args.tile[0]
        if num_frame_testing:
            # test as multiple clips if out-of-memory
            sf = args.scale
            num_frame_overlapping = args.tile_overlap[0]
            not_overlap_border = False
            b, d, c, h, w = lq.size()
            c = c - 1 if args.nonblind_denoising else c
            stride = num_frame_testing - num_frame_overlapping
            d_idx_list = list(range(0, d-num_frame_testing, stride)) + [max(0, d-num_frame_testing)]
            E = torch.zeros(b, d, c, h*sf, w*sf)
            W = torch.zeros(b, d, 1, 1, 1)

            for d_idx in d_idx_list:
                lq_clip = lq[:, d_idx:d_idx+num_frame_testing, ...]
                out_clip = test_clip(lq_clip, model, args)
                out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1))

                if not_overlap_border:
                    if d_idx < d_idx_list[-1]:
                        out_clip[:, -num_frame_overlapping//2:, ...] *= 0
                        out_clip_mask[:, -num_frame_overlapping//2:, ...] *= 0
                    if d_idx > d_idx_list[0]:
                        out_clip[:, :num_frame_overlapping//2, ...] *= 0
                        out_clip_mask[:, :num_frame_overlapping//2, ...] *= 0

                E[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip)
                W[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip_mask)
            output = E.div_(W)
        else:
            # test as one clip (the whole video) if you have enough memory
            window_size = args.window_size
            d_old = lq.size(1)
            d_pad = (window_size[0] - d_old % window_size[0]) % window_size[0]
            lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1) if d_pad else lq
            output = test_clip(lq, model, args)
            output = output[:, :d_old, :, :, :]

        return output


def test_clip(lq, model, args):
    ''' test the clip as a whole or as patches. '''

    sf = args.scale
    window_size = args.window_size
    size_patch_testing = args.tile[1]
    assert size_patch_testing % window_size[-1] == 0, 'testing patch size should be a multiple of window_size.'

    if size_patch_testing:
        # divide the clip to patches (spatially only, tested patch by patch)
        overlap_size = args.tile_overlap[1]
        not_overlap_border = True

        # test patch by patch
        b, d, c, h, w = lq.size()
        c = c - 1 if args.nonblind_denoising else c
        stride = size_patch_testing - overlap_size
        h_idx_list = list(range(0, h-size_patch_testing, stride)) + [max(0, h-size_patch_testing)]
        w_idx_list = list(range(0, w-size_patch_testing, stride)) + [max(0, w-size_patch_testing)]
        E = torch.zeros(b, d, c, h*sf, w*sf)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = lq[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing]
                out_patch,_,_,_ = model(in_patch)
                out_patch = out_patch.detach().cpu()
                out_patch_mask = torch.ones_like(out_patch)

                if not_overlap_border:
                    if h_idx < h_idx_list[-1]:
                        out_patch[..., -overlap_size//2:, :] *= 0
                        out_patch_mask[..., -overlap_size//2:, :] *= 0
                    if w_idx < w_idx_list[-1]:
                        out_patch[..., :, -overlap_size//2:] *= 0
                        out_patch_mask[..., :, -overlap_size//2:] *= 0
                    if h_idx > h_idx_list[0]:
                        out_patch[..., :overlap_size//2, :] *= 0
                        out_patch_mask[..., :overlap_size//2, :] *= 0
                    if w_idx > w_idx_list[0]:
                        out_patch[..., :, :overlap_size//2] *= 0
                        out_patch_mask[..., :, :overlap_size//2] *= 0

                E[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch_mask)
        output = E.div_(W)

    else:
        _, _, _, h_old, w_old = lq.size()
        h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]
        w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]

        lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3) if h_pad else lq
        lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4) if w_pad else lq

        output = model(lq).detach().cpu()

        output = output[0][:, :, :, :h_old*sf, :w_old*sf]

    return output


if __name__ == '__main__':
    main()
