import random
import numpy as np
import lmdb
import cv2
import torch
import torch.utils.data as data
import data.util as util



class AIM2020Dataset(data.Dataset):
    '''Read LQ images only in the test phase.'''

    def __init__(self, opt):
        super(AIM2020Dataset, self).__init__()

        self.opt = opt
        self.data_type= self.opt['data_type']
        self.crop_size = self.opt['crop_size']
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        self.paths_HQ, self.sizes_HQ = util.get_image_paths(self.data_type, opt['dataroot_HQ'])
        assert self.paths_LQ, 'Error: LQ paths are empty.'
        assert self.paths_HQ, 'Error: HQ paths are empty.'
    def __getitem__(self, index):


        # get LQ and HQ image
        LQ_path = self.paths_LQ[index]
        HQ_path = self.paths_HQ[index]

        img_LQ = util.read_img(None, LQ_path, None)
        img_HQ = util.read_img(None, HQ_path, None)

        if self.opt['crop_size']:
            # randomly crop
            LH, LW, _ = img_LQ.shape
            rnd_h = random.randint(0, max(0, LH - self.crop_size))
            rnd_w = random.randint(0, max(0, LW - self.crop_size))

            rnd_hh = rnd_h * self.opt['scale']
            rnd_wh = rnd_w * self.opt['scale']
            patch_size = self.crop_size * self.opt['scale']
            img_LQ = img_LQ[rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size, :]
            img_HQ = img_HQ[rnd_hh:rnd_hh + patch_size, rnd_wh:rnd_wh + patch_size, :]

        if self.opt['phase'] == 'train':
            # augmenttation cutclur
            if self.opt['use_cutblur']:
                img_LQ,img_HQ=util.argment_cutblur(img_LQ, img_HQ,self.opt['scale'])
            if self.opt['use_rgbpermute']:
                img_LQ, img_HQ=util.argment_rgb(img_LQ, img_HQ)
            # augmentation - flip, rotate
            img_LQ, img_HQ = util.augment([img_LQ, img_HQ], self.opt['use_flip'],
                                          self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_LQ.shape[2] == 3:
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        if img_HQ.shape[2] == 3:
            img_HQ = img_HQ[:, :, [2, 1, 0]]

        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        img_HQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ, (2, 0, 1)))).float()

        return {'LQ': img_LQ, 'LQ_path': LQ_path, 'HQ': img_HQ, 'HQ_path': HQ_path}

    def __len__(self):
        return len(self.paths_LQ)