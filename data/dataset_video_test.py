import os
import glob
import numpy as np
import torch
from os import path as osp
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import utils.utils_video as utils_video


class VideoRecurrentTestDataset(data.Dataset):

    def __init__(self, opt):
        super(VideoRecurrentTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.gt_size = opt.get('gt_size', [1024,1920])
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}

        self.imgs_lq, self.imgs_gt = {}, {}
        subfolders = ['001', '012', '018', '039', '044', '046', '062', '065', '070', '095', '102', '116', 
                 '143', '145', '164', '196', '207', '225', '263', '278', '288', '290', '365', '381', 
                 '435', '448', '465', '468', '485', '489', '527', '528', '540', '545', '552', '559', 
                 '581', '584', '592', '597', '602', '606', '614', '620', '628', '629', '635', '636', 
                 '653', '676']
        for subfolder in subfolders:
            img_paths_lq=[]
            img_paths_gt=[]
            for j in range(50):
                key = subfolder+'/'+str(j).zfill(3)+'.npy'
                img_paths_lq.append(key)
                img_paths_gt.append(key)
            max_idx = len(img_paths_lq)
            self.data_info['lq_path'].extend(img_paths_lq)
            self.data_info['gt_path'].extend(img_paths_gt)
            self.data_info['folder'].extend([subfolder] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append(f'{i}/{max_idx}')
            border_l = [0] * max_idx
            for i in range(self.opt['num_frame'] // 2):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            self.imgs_lq[subfolder] = img_paths_lq
            self.imgs_gt[subfolder] = img_paths_gt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        # Find unique folder strings
        self.folders = sorted(list(set(self.data_info['folder'])))

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
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        folder = self.folders[index]
        img_lqs = []
        img_gts = []
        for ind in range(50):
            img_lq_path = folder + '/' + str(ind).zfill(3) +'.npy'
            img_gt_path =folder + '/' + str(ind).zfill(3) +'.npy'

            img_lq = self.file_client.get(os.path.join(self.lq_root, 'Input',img_lq_path), 'lq')
            img_gt = self.file_client.get(os.path.join(self.gt_root, 'GT',img_gt_path), 'gt')
            img_lq = self._tonemap(img_lq)
            img_gt = self._tonemap(img_gt)

            img_lqs.append(img_lq)
            img_gts.append(img_gt)
        img_gts, img_lqs = utils_video.paired_center_crop(img_gts, img_lqs, self.gt_size)
        img_lqs = utils_video.img2tensor(img_lqs,bgr2rgb=False,float32=True)
        img_gts = utils_video.img2tensor(img_gts,bgr2rgb=False,float32=True)
        img_lqs = torch.stack(img_lqs, dim=0)
        img_gts = torch.stack(img_gts, dim=0)
        img_gts_L1 = F.interpolate(img_gts, scale_factor=0.5, mode='bilinear', align_corners=False)
        img_gts_L2 = F.interpolate(img_gts_L1, scale_factor=0.5, mode='bilinear', align_corners=False)
        img_gts_L3 = F.interpolate(img_gts_L2, scale_factor=0.5, mode='bilinear', align_corners=False)
        return {
            'L': img_lqs,
            'H': img_gts,
            'H1': img_gts_L1,
            'H2': img_gts_L2,
            'H3': img_gts_L3,
            'folder': folder,
            'lq_path': self.imgs_lq[folder],
        }

    def __len__(self):
        return len(self.folders)
