# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import os
import warnings
import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torchvision import utils as vutils
from torch.nn.modules.batchnorm import _BatchNorm
from distutils.version import LooseVersion
from torch.nn.modules.utils import _pair, _single
import numpy as np
from functools import reduce, lru_cache
from operator import mul
import pdb


def kaiming_init(module,a = 0.0, mode= 'fan_out',nonlinearity = 'relu',bias = 0.0,distribution= 'normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def default_init_weights(module, scale=1):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, _BatchNorm):
            constant_init(m.weight, val=1, bias=0)


class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)



def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True, use_pad_mask=False):

    # assert x.size()[-2:] == flow.size()[1:3] # temporaily turned off for image-wise shift
    n, _, h, w = x.size()
    # create mesh grid
    # grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x)) # an illegal memory access on TITAN RTX + PyTorch1.9.1
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, dtype=x.dtype, device=x.device), torch.arange(0, w, dtype=x.dtype, device=x.device))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow

    # if use_pad_mask: # for PWCNet
    #     x = F.pad(x, (0,0,0,0,0,1), mode='constant', value=1)

    # scale grid to [-1,1]
    if interp_mode == 'nearest4': # todo: bug, no gradient for flow model in this case!!! but the result is good
        vgrid_x_floor = 2.0 * torch.floor(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_x_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_y_floor = 2.0 * torch.floor(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0
        vgrid_y_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0

        output00 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_floor), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output01 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_ceil), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output10 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_floor), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output11 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_ceil), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)

        return torch.cat([output00, output01, output10, output11], 1)

    else:
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

        # if use_pad_mask: # for PWCNet
        #     output = _flow_warp_masking(output)

        # TODO, what if align_corners=False
        return output


class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
        return_levels (list[int]): return flows of different levels. Default: [5].
    """
    def __init__(self, load_path=None, return_levels=[3,4,5]):
        super(SpyNet, self).__init__()
        self.return_levels = return_levels
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if load_path:
            if not os.path.exists(load_path):
                import requests
                url = 'https://github.com/JingyunLiang/VRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth'
                r = requests.get(url, allow_redirects=True)
                print(f'downloading SpyNet pretrained model from {url}')
                os.makedirs(os.path.dirname(load_path), exist_ok=True)
                open(load_path, 'wb').write(r.content)

            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp, w, h, w_floor, h_floor):
        flow_list = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        flow = ref[0].new_zeros(
            [ref[0].size(0), 2,
             int(math.floor(ref[0].size(2) / 2.0)),
             int(math.floor(ref[0].size(3) / 2.0))])

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(
                    supp[level], upsampled_flow.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border'),
                upsampled_flow
            ], 1)) + upsampled_flow

            if level in self.return_levels:
                scale = 2**(5-level) # level=5 (scale=1), level=4 (scale=2), level=3 (scale=4), level=2 (scale=8)
                flow_out = F.interpolate(input=flow, size=(h//scale, w//scale), mode='bilinear', align_corners=False)
                flow_out[:, 0, :, :] *= float(w//scale) / float(w_floor//scale)
                flow_out[:, 1, :, :] *= float(h//scale) / float(h_floor//scale)
                flow_list.insert(0, flow_out)

        return flow_list

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        flow_list = self.process(ref, supp, w, h, w_floor, h_floor)

        return flow_list[0] if len(flow_list) == 1 else flow_list


class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels=64, res_scale=1.0, groups=1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(
            mid_channels, mid_channels, 3, 1, 1, bias=True, groups=groups)
        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, 3, 1, 1, bias=True, groups=groups)

        self.relu = nn.ReLU(inplace=True)

        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


def make_layer(block, num_blocks, **kwarg):
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()
        main = []
        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))
        self.main = nn.Sequential(*main)
    def forward(self, feat):
        return self.main(feat)


class PixelShufflePack(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        default_init_weights(self, 1)

    def forward(self, x):
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


class DAM(nn.Module):
    ##Decoupling Attention Modules
    def __init__(self, in_channels=64, out_channels=64, num_blocks=5):
        super().__init__()

        self.dehaze_conv0 =  nn.Conv2d(out_channels, 3, 3, 1, 1, bias=False)
        self.deflare_conv0 =  nn.Conv2d(out_channels, 3, 3, 1, 1, bias=False)

        self.dehaze_conv1 =  nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.deflare_conv1 =  nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)

        self.dehaze_conv_act =  nn.Conv2d(3, out_channels, 3, 1, 1, bias=False)
        self.deflare_conv_act =  nn.Conv2d(3, out_channels, 3, 1, 1, bias=False)

        RefineBlock = []
        RefineBlock.append(nn.Conv2d(in_channels + out_channels * 2, out_channels, 3, 1, 1, bias=True))
        RefineBlock.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        RefineBlock.append(
            make_layer(
                ResidualBlockNoBN, num_blocks , mid_channels=out_channels))
        self.RefineBlock = nn.Sequential(*RefineBlock)

        self.conv_out =  nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)

    def forward(self, feat_current, feat_prev, feat_prevT,  feat_prop, flare_map, img):

        haze_map = 1-flare_map
        img_out = img  + self.dehaze_conv0(haze_map * feat_prevT)  + self.deflare_conv0(flare_map * feat_prop)

        dehaze_feat =  self.dehaze_conv1(haze_map * feat_prevT) 
        deflare_feat =  self.deflare_conv1(flare_map * feat_prop) 

        dehaze_act_feat = torch.sigmoid(self.dehaze_conv_act(img_out))
        deflare_act_feat = torch.sigmoid(self.deflare_conv_act(img_out))

        dehaze_feat = dehaze_act_feat * dehaze_feat + feat_prevT
        deflare_feat = deflare_act_feat * deflare_feat + feat_prop

        feat_current = torch.cat([feat_current, dehaze_feat, deflare_feat],dim=1)

        feat_current = self.RefineBlock(feat_current)

        out_feat = self.conv_out(feat_current)
        return out_feat,  feat_current, img_out



class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor=0):
        super().__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))
    def forward(self, x):
        x = self.down(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor=0):
        super().__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))
    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

class Feat_Encoder(nn.Module):
    def __init__(self, channels=64, scale_unetfeats=8):
        super().__init__()
        self.encoder_level0 = nn.Sequential(
            nn.Conv2d(3, channels, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.encoder_level1 = ResidualBlocksWithInputConv(channels, channels, 2)
        self.encoder_level2 = ResidualBlocksWithInputConv(channels + scale_unetfeats, channels + scale_unetfeats, 2)
        self.encoder_level3 = ResidualBlocksWithInputConv(channels + scale_unetfeats*2, channels + scale_unetfeats*2,2)

        self.down12 = DownSample(channels, scale_unetfeats)
        self.down23 = DownSample(channels + scale_unetfeats, scale_unetfeats)

    def forward(self, x):
        x = self.encoder_level0(x) 
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)
        return enc1, enc2, enc3

class Feat_Decoder(nn.Module):
    def __init__(self, channels=64, scale_unetfeats=8):
        super().__init__()

        self.decoder_level1 = ResidualBlocksWithInputConv(channels, channels, 2)
        self.decoder_level2 = ResidualBlocksWithInputConv(channels + scale_unetfeats, channels + scale_unetfeats, 2)
        self.decoder_level3 = ResidualBlocksWithInputConv(channels + scale_unetfeats*2, channels + scale_unetfeats*2,2)

        self.skip_attn1 = ResidualBlockNoBN(channels)
        self.skip_attn2 = ResidualBlockNoBN(channels + scale_unetfeats)

        self.up21 = SkipUpSample(channels, scale_unetfeats)
        self.up32 = SkipUpSample(channels + scale_unetfeats, scale_unetfeats)

    def forward(self, enc1, enc2, enc3):
        dec3 = self.decoder_level3(enc3)
        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)
        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)
        return dec1


class Img_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
    def forward(self, x):
        x1 = self.down(x)
        x2 = self.down(x1)
        x3 = self.down(x2)
        return x1,x2, x3


class DDRNet(nn.Module):

    def __init__(self,
                 mid_channels=48,
                 scale_feat=12,
                 num_blocks=5,
                 max_residue_magnitude=10,
                 tau = 0.95,
                 spynet_pretrained=None,
                 use_flow=True):
        super().__init__()
        self.mid_channels = mid_channels
        self.use_flow = use_flow
        self.scale_feat = scale_feat
        self.tau = tau
        # optical flow
        self.spynet = SpyNet(load_path=spynet_pretrained, return_levels=[3,4,5])

        # feature extraction module
        self.Feat_Encoder = Feat_Encoder(channels=mid_channels, scale_unetfeats= scale_feat)
        self.Img_Encoder = Img_Encoder()
        self.Max_pool = nn.MaxPool2d(kernel_size=5, stride=2,padding=2)

        # propagation branches
        self.DAM_level1 = nn.ModuleDict()
        self.DAM_level2 = nn.ModuleDict()
        self.DAM_level3 = nn.ModuleDict()
        modules = ['backward_1', 'forward_1']
        for i, module in enumerate(modules):
            self.DAM_level1[module] = DAM((1 + i) * mid_channels, mid_channels, num_blocks)
            self.DAM_level2[module] = DAM((1 + i) * (mid_channels+scale_feat), mid_channels+scale_feat, num_blocks)
            self.DAM_level3[module] = DAM((1 + i) * (mid_channels+scale_feat*2), mid_channels+scale_feat*2, num_blocks)

        self.Feat_Decoder = Feat_Decoder(channels=mid_channels, scale_unetfeats= scale_feat)

        # upsampling module
        self.upsample1 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    
    def compute_flare(self,frame,tau = 0.95,save = False):
        max_c = frame.max(dim = 1,keepdim=True).values - tau
        max_c[max_c<0] = 0
        alpha = (max_c / (1 - tau)).float()
        if save==True:
            vutils.save_image(alpha[0],'./flare_map.png')
            vutils.save_image(frame[0],'./image_map.png')
        map_1 = self.Max_pool(alpha)
        map_2 = self.Max_pool(map_1)
        map_3 = self.Max_pool(map_2)
        return map_1,map_2, map_3

    def compute_flow(self, lqs):
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2)
        flows_forward = self.spynet(lqs_2, lqs_1)

        return flows_forward, flows_backward

    def propagate(self,lqs,feats, maps, imgs, module_name, flows=None):
        n, t, _, h, w = lqs.size()
        frame_idx = range(0, t)
        flow_idx = range(-1, t-1)
        mapping_idx = list(range(0, len(feats['level_1'])))
        mapping_idx += mapping_idx[::-1]
        prev_id = -1

        feat_set_1 = feats['backward_1_level_1']
        feat_set_2 = feats['backward_1_level_2']
        feat_set_3 = feats['backward_1_level_3']

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx
            prev_id = 1

            feat_set_1 = feats['level_1']
            feat_set_2 = feats['level_2']
            feat_set_3 = feats['level_3']

        feat_prop_1 = lqs.new_zeros(n, self.mid_channels, h//2, w//2)
        feat_prop_2 = feat_prop_1.new_zeros(n, self.mid_channels+self.scale_feat, h//4, w//4)
        feat_prop_3 = feat_prop_1.new_zeros(n, self.mid_channels+self.scale_feat*2, h//8, w//8)

        for i, idx in enumerate(frame_idx):
            feat_current_1 = feat_set_1[mapping_idx[idx]]
            feat_current_2 = feat_set_2[mapping_idx[idx]]
            feat_current_3 = feat_set_3[mapping_idx[idx]]
            map_1 = maps['level_1'][mapping_idx[idx]]
            map_2 = maps['level_2'][mapping_idx[idx]]
            map_3 = maps['level_3'][mapping_idx[idx]]
            img_1 = imgs['level_1'][mapping_idx[idx]]
            img_2 = imgs['level_2'][mapping_idx[idx]]
            img_3 = imgs['level_3'][mapping_idx[idx]]
            feat_prev_1 = feat_current_1
            feat_prev_2 = feat_current_2
            feat_prev_3 = feat_current_3
            feat_prevT_1 = feat_current_1
            feat_prevT_2 = feat_current_2
            feat_prevT_3 = feat_current_3

            if i > 0:
                feat_prev_1 = feat_set_1[mapping_idx[idx+prev_id]]
                feat_prev_2 = feat_set_2[mapping_idx[idx+prev_id]]
                feat_prev_3 = feat_set_3[mapping_idx[idx+prev_id]]
                feat_prop_1 = torch.cat([feat_prev_1,feat_prop_1],dim=1)
                feat_prop_2 = torch.cat([feat_prev_2,feat_prop_2],dim=1)
                feat_prop_3 = torch.cat([feat_prev_3,feat_prop_3],dim=1)
                if flows:
                    flow_1 = flows['level_1'][:, flow_idx[i], :, :, :]
                    flow_2 = flows['level_2'][:, flow_idx[i], :, :, :]
                    flow_3 = flows['level_3'][:, flow_idx[i], :, :, :]
                feat_prop_1 = flow_warp(feat_prop_1, flow_1.permute(0, 2, 3, 1))
                feat_prop_2 = flow_warp(feat_prop_2, flow_2.permute(0, 2, 3, 1))
                feat_prop_3 = flow_warp(feat_prop_3, flow_3.permute(0, 2, 3, 1))
                feat_prevT_1,  feat_prop_1= torch.chunk(feat_prop_1, 2,dim=1)
                feat_prevT_2,  feat_prop_2= torch.chunk(feat_prop_2, 2,dim=1)
                feat_prevT_3,  feat_prop_3= torch.chunk(feat_prop_3, 2,dim=1)


            if 'forward' in module_name:
                feat_current_1 = torch.cat([feats['level_1'][idx],feat_current_1],dim=1)
                feat_current_2 = torch.cat([feats['level_2'][idx],feat_current_2],dim=1)
                feat_current_3 = torch.cat([feats['level_3'][idx],feat_current_3],dim=1)

            feat_current_1, feat_prop_1, img_out_1 =  self.DAM_level1[module_name](feat_current_1, feat_prev_1, feat_prevT_1, feat_prop_1, map_1, img_1)
            feat_current_2, feat_prop_2, img_out_2 =  self.DAM_level2[module_name](feat_current_2, feat_prev_2, feat_prevT_2, feat_prop_2, map_2, img_2)
            feat_current_3, feat_prop_3, img_out_3 =  self.DAM_level3[module_name](feat_current_3, feat_prev_3, feat_prevT_3, feat_prop_3, map_3, img_3)

            feats[module_name+'_level_1'].append(feat_current_1)
            feats[module_name+'_level_2'].append(feat_current_2)
            feats[module_name+'_level_3'].append(feat_current_3)

            imgs[module_name+'_level_1'].append(img_out_1)
            imgs[module_name+'_level_2'].append(img_out_2)
            imgs[module_name+'_level_3'].append(img_out_3)

        if 'backward' in module_name:
            feats[module_name+'_level_1'] = feats[module_name+'_level_1'][::-1]
            feats[module_name+'_level_2'] = feats[module_name+'_level_2'][::-1]
            feats[module_name+'_level_3'] = feats[module_name+'_level_3'][::-1]

            imgs[module_name+'_level_1'] = imgs[module_name+'_level_1'][::-1]
            imgs[module_name+'_level_2'] = imgs[module_name+'_level_2'][::-1]
            imgs[module_name+'_level_3'] = imgs[module_name+'_level_3'][::-1]

        return feats,imgs

    def upsample(self, lqs, feats):
        outputs = []
        num_outputs = len(feats)

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = self.lrelu(self.upsample1(feats[i]))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            hr += lqs[:, i, :, :, :]
            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs):

        n, t, c, h, w = lqs.size()

        # compute optical flow using the low-res inputs
        if self.use_flow:
            lqs_downsample = F.interpolate(lqs.view(-1, c, h, w), scale_factor=0.5,mode='bicubic').view(n, t, c, h // 2, w // 2)
            assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
                'The height and width of low-res inputs must be at least 64, '
                f'but got {h} and {w}.')
            flows_forward, flows_backward = self.compute_flow(lqs_downsample) 

        feats_1,feats_2,feats_3 = self.Feat_Encoder(lqs.view(-1, c, h, w))
        imgs_1,imgs_2,imgs_3 = self.Img_Encoder(lqs.view(-1, c, h, w))

        feats_1 = feats_1.view(n, t, -1, h // 2, w // 2) # n * t * 64 * 128 * 128
        feats_2 = feats_2.view(n, t, -1, h // 4, w // 4) # n * t * 72 * 64 * 64
        feats_3 = feats_3.view(n, t, -1, h // 8, w // 8) # n * t * 80 * 32 * 32

        imgs_1 = imgs_1.view(n, t, -1,  h // 2, w // 2) # n * t * 3 * 128 * 128
        imgs_2 = imgs_2.view(n, t, -1,  h // 4, w // 4) # n * t * 3 * 64 * 64
        imgs_3 = imgs_3.view(n, t, -1,  h // 8, w // 8) # n * t * 3 * 32 * 32

        maps_1,  maps_2, maps_3= self.compute_flare(lqs.view(-1, c, h, w),self.tau)
        
        feats = {}
        feats['level_1'] = [feats_1[:, i, :, :, :] for i in range(0, t)]
        feats['level_2'] = [feats_2[:, i, :, :, :] for i in range(0, t)]
        feats['level_3'] = [feats_3[:, i, :, :, :] for i in range(0, t)]

        imgs = {}
        imgs['level_1'] = [imgs_1[:, i, :, :, :] for i in range(0, t)]
        imgs['level_2'] = [imgs_2[:, i, :, :, :] for i in range(0, t)]
        imgs['level_3'] = [imgs_3[:, i, :, :, :] for i in range(0, t)]

        maps = {}
        maps['level_1'] = [maps_1.view(n,t,1,h // 2, w // 2)[:, i, :, :, :] for i in range(0, t)]
        maps['level_2'] = [maps_2.view(n,t,1,h // 4, w // 4)[:, i, :, :, :] for i in range(0, t)]
        maps['level_3'] = [maps_3.view(n,t,1,h // 8, w // 8)[:, i, :, :, :] for i in range(0, t)]

        # feature propagation
        for iter_ in [1]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'
                feats[module+'_level_1'] = []
                feats[module+'_level_2'] = []
                feats[module+'_level_3'] = []
                imgs[module+'_level_1'] = []
                imgs[module+'_level_2'] = []
                imgs[module+'_level_3'] = []
                flows = {}
                if direction == 'backward':
                    flows['level_1'] = flows_backward[0].view(n,t-1,2,h // 2, w // 2)
                    flows['level_2'] = flows_backward[1].view(n,t-1,2,h // 4, w // 4)
                    flows['level_3'] = flows_backward[2].view(n,t-1,2,h // 8, w // 8)
                else:
                    flows['level_1'] = flows_forward[0].view(n,t-1,2,h // 2, w // 2)
                    flows['level_2'] = flows_forward[1].view(n,t-1,2,h // 4, w // 4)
                    flows['level_3'] = flows_forward[2].view(n,t-1,2,h // 8, w // 8)
                feats,imgs = self.propagate(lqs, feats, maps,  imgs, module, flows)

        feats_1 = torch.cat(feats['forward_1_level_1'],dim=0)
        feats_2 = torch.cat(feats['forward_1_level_2'],dim=0)
        feats_3 = torch.cat(feats['forward_1_level_3'],dim=0)

        feats_out = self.Feat_Decoder(feats_1,feats_2,feats_3)
        feats_out = torch.chunk(feats_out,t,dim=0)
        return self.upsample(lqs, feats_out),torch.stack(imgs['backward_1_level_1'], dim=1),torch.stack(imgs['forward_1_level_2'], dim=1),torch.stack(imgs['forward_1_level_3'], dim=1)
