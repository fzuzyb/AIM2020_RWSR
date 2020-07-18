import sys
sys.path.append('../')
import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict
from tqdm import tqdm
import options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model
from thop import profile, clever_format
import os
import torch
import cv2


def get_lqs(test_input_dir,frame,PAD):
    img = cv2.imread(os.path.join(test_input_dir, frame), cv2.IMREAD_UNCHANGED)
    #HWC

    imgpachs=util.sep_iter_data(img,PAD)

    img_tensors = [util.img2tensor(img_pach, None, True, is_yuv=False).unsqueeze(0) for img_pach in imgpachs]

    return img_tensors,img.shape

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
args=parser.parse_args()
opt = option.parse(args.opt, is_train=False)
opt = option.dict_to_nonedict(opt)



util.mkdirs(opt['path']['results_root'])
#set log
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)

logger = logging.getLogger('base')
logger.info(option.dict2str(opt))
pad=opt['datasets']['val']['pad']
scale=opt['datasets']['val']['scale']
ensemble=opt['datasets']['val']['ensemble']
#### Create test dataset and dataloader
# test_loaders = []
# for phase, dataset_opt in sorted(opt['datasets'].items()):
#     test_set = create_dataset(dataset_opt)
#     test_loader = create_dataloader(test_set, dataset_opt)
#     logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
#     test_loaders.append(test_loader)

model = create_model(opt)

img_list=sorted(os.listdir(opt['datasets']['val']['dataroot_LQ']))
#cal flops and params
# input = torch.randn(1, 3, 512, 512).to(model.device)
# flops, params = profile(model.get_network().module, inputs=(input, ))
# flops, params = clever_format([flops, params], "%.3f")
# print('flops={}, params={}'.format(flops, params))

test_set_name = opt['datasets']['val']['name']
logger.info('\nTesting [{:s}]...'.format(test_set_name))
test_start_time = time.time()
dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
t=tqdm(range(len(img_list)))
if ensemble:
    network=model.get_network()
    network.eval()

for idx,img_name in enumerate(img_list):

    # model.feed_data(data)
    img_tensors,img_shape=get_lqs(opt['datasets']['val']['dataroot_LQ'],img_name,pad)

    if ensemble:
        with torch.no_grad():
            sr_imgs=[util.flipx4_forward(network,img_tensor) for img_tensor in img_tensors]

    else:

        sr_imgs=[model.test(img_tensor) for img_tensor in img_tensors]


    sr_imgs = [util.tensor2img(sr_img) for sr_img in sr_imgs] # uint8

    sr_shape = [x*scale for x in img_shape[:-1]]

    if img_shape[0]%2==0:
        biash=scale
    else:
        biash=scale-2
    if img_shape[1]%2==0:
        biasw=scale
    else:
        biasw=scale-2
    sr_img = util.mergeimg(sr_imgs, sr_shape, pad, scale,biash,biasw)


    assert [x * scale for x in img_shape[:-1]] == [x for x in sr_img.shape[:-1]], print("sr size does not correct")
    # save images
    util.mkdir(dataset_dir)
    save_img_path = osp.join(dataset_dir,img_name)

    util.save_img(sr_img, save_img_path)

    t.update()