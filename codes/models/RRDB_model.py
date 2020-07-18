
import sys
sys.path.append('../')
import logging
from collections import OrderedDict
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from .base_model import BaseModel


import utils.util as util
from .__init__ import define_SR
import torch.nn.functional as F
logger = logging.getLogger('base')


class RRDBM(BaseModel):
    def __init__(self, opt):
        super(RRDBM, self).__init__(opt)



        # define networks and load pretrained models
        train_opt = opt['train']

        self.netG_R = define_SR(opt).to(self.device)

        if opt['dist']:
            self.netG_R = DistributedDataParallel(self.netG_R, device_ids=[torch.cuda.current_device()])

        else:
            self.netG_R = DataParallel(self.netG_R)
        # define losses, optimizer and scheduler
        if self.is_train:
            # losses
            # if train_opt['l_pixel_type']=="L1":
            #     self.criterionPixel= torch.nn.L1Loss().to(self.device)
            # elif train_opt['l_pixel_type']=="CR":
            #     self.criterionPixel=CharbonnierLoss().to(self.device)
            #
            # else:
            #     raise NotImplementedError("pixel_type does not implement still")
            self.criterionPixel=SRLoss(loss_type=train_opt['l_pixel_type']).to(self.device)
            # optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_R.parameters(),
                                                lr=train_opt['lr'], betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)


            #scheduler
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError("lr_scheme does not implement still")


            self.log_dict = OrderedDict()
            self.train_state()


        self.load()  # load R


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    def feed_data(self, data):
        self.LQ = data['LQ'].to(self.device)
        self.HQ = data['HQ'].to(self.device)


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.fake_HQ = self.netG_R(self.LQ)


    def backward_G(self,step):
        """Calculate the loss for generators G_A and G_B"""


        self.loss_G_pixel = self.criterionPixel(self.fake_HQ, self.HQ)
        if len(self.loss_G_pixel)==2:
            if self.opt['train']['other_step'] < step:
                self.loss_G_total = self.loss_G_pixel[0] * self.opt['train']['l_l1_weight']+ \
                                    self.loss_G_pixel[1] * self.opt['train']['l_ssim_weight']
            else:
                self.loss_G_total = self.loss_G_pixel[0] * self.opt['train']['l_l1_weight']
        else:

            self.loss_G_total=self.loss_G_pixel[0]*self.opt['train']['l_l1_weight']


        self.loss_G_total.backward()


    def optimize_parameters(self,step):
        # G
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G
        self.optimizer_G.zero_grad()  # set G gradients to zero
        self.backward_G(step)  # calculate gradients for G
        self.optimizer_G.step()  # update G's weights


        # set log
        for i in range(len(self.loss_G_pixel)):
            self.log_dict[str(i)] = self.loss_G_pixel[i].item()
        # self.log_dict['loss_l1'] = self.loss_G_pixel.item() if self.opt['train']['l_l1_weight']!=0 else 0

    def train_state(self):
        self.netG_R.train()



    def test_state(self):
        self.netG_R.eval()


    def val(self):
        self.test_state()
        with torch.no_grad():
            self.forward()
        self.train_state()

    def test(self,img):

        self.netG_R.eval()
        with torch.no_grad():

            SR=self.netG_R(img)
        return SR

    def get_network(self):
        return self.netG_R

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals_and_cal_metric(self,opt,current_step):

        visuals = [F.interpolate(self.LQ, scale_factor=self.opt['datasets']['train']['scale'], mode='bilinear', align_corners=True),
                   self.fake_HQ,
                   self.HQ]


        util.write_2images(visuals, opt['datasets']['val']['batch_size'], opt['path']['val_images'],
                      'test_%08d' % (current_step))

        # HTML
        util.write_html(opt['path']['experiments_root'] + "/index.html", (current_step), opt['train']['val_freq'], opt['path']['val_images'])

        #src BRG range [0-255] HWC
        srimg = util.tensor2img(self.fake_HQ)
        hrimg = util.tensor2img(self.HQ)


        psnr = calculate_psnr(srimg, hrimg)
        ssim = calculate_ssim(srimg, hrimg)
        return {"psnr": psnr, "ssim": ssim}

    def print_network(self):


        if self.is_train:
            # Generator
            s, n = self.get_network_description(self.netG_R)
            net_struc_str = '{} - {}'.format(self.netG_R.__class__.__name__,
                                             self.netG_R.module.__class__.__name__)
            logger.info('Network G_R structure: {}, with parameters: {:,d}'.format(
                net_struc_str, n))
            logger.info(s)



    def load(self):
        load_path_G_R = self.opt['path']['pretrain_model_G_R']

        if load_path_G_R is not None :
            logger.info('Loading models for G [{:s}] ...'.format(load_path_G_R))
            self.load_network(load_path_G_R, self.netG_R, self.opt['path']['strict_load'])




    def save(self, iter_step):
        self.save_network(self.netG_R, 'G_R', iter_step)


