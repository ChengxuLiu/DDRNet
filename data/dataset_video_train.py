import os
import numpy as np
import random
import torch
from pathlib import Path
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F
import utils.utils_video as utils_video


class VideoRecurrentTrainDataset(data.Dataset):
    def __init__(self, opt):
        super(VideoRecurrentTrainDataset, self).__init__()
        self.opt = opt
        self.scale = opt.get('scale', 1)
        self.gt_size = opt.get('gt_size', 256)
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.filename_tmpl = opt.get('filename_tmpl', '03d')
        self.filename_ext = opt.get('filename_ext', 'npy')
        self.num_frame = opt['num_frame']
        keys = []
        total_num_frames = [] # some clips may not have 100 frames
        start_frames = [] # some clips may not start from 00000
        for folder in range(677):
            keys.extend([f'{folder:{self.filename_tmpl}}'])
            total_num_frames.extend([50])
            start_frames.extend([0])
        # remove the video clips used in validation
        val_partition = ['001', '012', '018', '039', '044', '046', '062', '065', '070', '095', '102', '116', 
        '143', '145', '164', '196', '207', '225', '263', '278', '288', '290', '365', '381', 
        '435', '448', '465', '468', '485', '489', '527', '528', '540', '545', '552', '559', 
        '581', '584', '592', '597', '602', '606', '614', '620', '628', '629', '635', '636', 
        '653', '676']
        self.keys = []
        self.total_num_frames = []
        self.start_frames = []
        if opt['test_mode']:
            for i, v in zip(range(len(keys)), keys):
                if v in val_partition:
                    self.keys.append(keys[i])
                    self.total_num_frames.append(total_num_frames[i])
                    self.start_frames.append(start_frames[i])
        else:
            for i, v in zip(range(len(keys)), keys):
                if v not in val_partition:
                    self.keys.append(keys[i])
                    self.total_num_frames.append(total_num_frames[i])
                    self.start_frames.append(start_frames[i])

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        print(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')


    def _tonemap(self, x, type='simple'):
        x = x.astype(np.float32)
        if type == 'mu_law':
            norm_x = x / x.max()
            mapped_x = np.log(1 + 10000 * norm_x) / np.log(1 + 10000)
        elif type == 'simple':
            mapped_x = x / (x + 0.25)
        elif type == 'same':
            mapped_x = x
        else:
            raise NotImplementedError('tone mapping type [{:s}] is not recognized.'.format(type))
        return mapped_x

    def __getitem__(self, index):
        key = self.keys[index]
        total_num_frames = self.total_num_frames[index]
        start_frames = self.start_frames[index]

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = random.randint(0, total_num_frames- self.num_frame)
        endmost_start_frame_idx = start_frames + total_num_frames - self.num_frame * interval
        if start_frame_idx > endmost_start_frame_idx:
            start_frame_idx = random.randint(start_frames, endmost_start_frame_idx)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))
        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        for neighbor in neighbor_list:
            # print(neighbor)

            img_lq_path = key +  f'/{neighbor:{self.filename_tmpl}}' + f'.{self.filename_ext}'
            img_gt_path = key + f'/{neighbor:{self.filename_tmpl}}' + f'.{self.filename_ext}'
            # get LQ
            img_lq = np.load(os.path.join(self.lq_root, 'Input',img_lq_path))
            img_lq = self._tonemap(img_lq)
            img_lqs.append(img_lq)

            # get GT
            img_gt = np.load(os.path.join(self.gt_root, 'GT',img_gt_path))
            img_gt = self._tonemap(img_gt)
            img_gts.append(img_gt)

        # randomly crop
        img_gts, img_lqs = utils_video.paired_random_crop(img_gts, img_lqs, self.gt_size, self.scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = utils_video.augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])
        img_results = utils_video.img2tensor(img_results,bgr2rgb=False,float32=True)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)
        img_gts_L1 = F.interpolate(img_gts, scale_factor=0.5, mode='bilinear', align_corners=False)
        img_gts_L2 = F.interpolate(img_gts_L1, scale_factor=0.5, mode='bilinear', align_corners=False)
        img_gts_L3 = F.interpolate(img_gts_L2, scale_factor=0.5, mode='bilinear', align_corners=False)

        return {'L': img_lqs, 'H': img_gts,  'H1': img_gts_L1,  'H2': img_gts_L2,  'H3': img_gts_L3, 'key': key}

    def __len__(self):
        return len(self.keys)
