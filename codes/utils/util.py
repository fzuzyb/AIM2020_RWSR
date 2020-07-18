import os
import sys
import time
import math
import torch.nn.functional as F
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid
from shutil import get_terminal_size
import torchvision.utils as vutils
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


####################
# image convert
####################
def crop_border(img_list, crop_border):
    """Crop borders of images
    Args:
        img_list (list [Numpy]): HWC
        crop_border (int): crop border for each end of height and weight

    Returns:
        (list [Numpy]): cropped image list
    """
    if crop_border == 0:
        return img_list
    else:
        return [v[crop_border:-crop_border, crop_border:-crop_border] for v in img_list]


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path):
    cv2.imwrite(img_path, img)


def DUF_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code

    Args:
        x (Tensor, [B, T, C, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    B, T, C, H, W = x.size()
    x = x.view(-1, 1, H, W)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0
    if scale == 3:
        r_h = 3 - (H % 3)
        r_w = 3 - (W % 3)
    x = F.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], 'reflect')

    gaussian_filter = torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(B, T, C, x.size(2), x.size(3))
    return x


def single_forward(model, inp):
    """PyTorch model forward (single test), it is just a simple warpper
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model

    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    with torch.no_grad():
        model_output = model(inp)
        if isinstance(model_output, list) or isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
    output = output.data.float().cpu()
    return output


def flipx4_forward(model, inp):
    """Flip testing with X4 self ensemble, i.e., normal, flip H, flip W, flip H and W
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model

    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    # normal
    output_f = single_forward(model, inp)

    # flip W
    output = single_forward(model, torch.flip(inp, (-1, )))
    output_f = output_f + torch.flip(output, (-1, ))
    # flip H
    output = single_forward(model, torch.flip(inp, (-2, )))
    output_f = output_f + torch.flip(output, (-2, ))
    # flip both H and W
    output = single_forward(model, torch.flip(inp, (-2, -1)))
    output_f = output_f + torch.flip(output, (-2, -1))

    return output_f / 4


####################
# metric
####################


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()



class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

def __write_images(image_outputs, display_image_num, file_name):

    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=2, normalize=False)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, image_directory, postfix):

    __write_images(image_outputs, display_image_num, '%s/%s.png' % (image_directory, postfix))
def write_one_row_html(html_file, iterations, img_filename, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (iterations,img_filename.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (img_filename, img_filename, all_size))
    return
def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    write_one_row_html(html_file, iterations, '%s/test_%08d.png' % (image_directory,iterations), all_size)

    for j in range(iterations, image_save_iterations-1, -1):
        if j % image_save_iterations == 0:
            write_one_row_html(html_file, j, '%s/test_%08d.png' % (image_directory, j), all_size)
    html_file.write("</body></html>")
    html_file.close()

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 19: # CelebAMask-HQ
        cmap = np.array([(0,  0,  0), (204, 0,  0), (76, 153, 0),
                     (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
                     (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
                     (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153),
                     (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)],
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=19):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def tensor2label(label_tensor, n_label):

    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    # label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    label_numpy = label_tensor.numpy().astype(np.float32)
    label_numpy = label_numpy / 255.0

    return label_numpy


def generate_label_img(inputs, imsize):
    pred_batch = []
    for input in inputs:
        input = input.view(1, 19, imsize, imsize)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)

    label_batch = []
    for p in pred_batch:
        p = p.view(1, imsize, imsize)
        label_batch.append(tensor2label(p, 19))

    label_batch = np.array(label_batch)
    label_batch = torch.from_numpy(label_batch).cuda()

    return label_batch

def generate_label_img_nchannel(inputs):
    size = inputs.size()
    imsize=size[1]
    oneHot_size = (size[0], 19, size[1], size[2])
    labels_real = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    #dim index src
    labels_real = labels_real.scatter_(1, inputs.view(size[0], 1, size[1], size[2]).data, 1.0)

    pred_batch = []
    for input in labels_real:
        input = input.view(1, 19, imsize, imsize)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)

    label_batch = []
    for p in pred_batch:
        p = p.view(1, imsize, imsize)
        label_batch.append(tensor2label(p, 19))

    label_batch = np.array(label_batch)
    label_batch = torch.from_numpy(label_batch)

    return label_batch

def img2tensor(img,resize=False,normal=True,is_yuv=False):
    if not is_yuv:
        if normal:
            img = img.astype(np.float32) / 255.
    if resize:
        img=cv2.resize(img,(resize,resize))
    #brg to rgb
    if not is_yuv:
        img=img[:, :, [2, 1, 0]]
    else:
        img=bgr2yuv(img)   #bgr2img
    #hwc to chw
    tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()

    return tensor
def label2tensor(label,resize=False,normal=True):
    if normal:
        label = label.astype(np.float32) / 255.
    if resize:
        label = cv2.resize(label, (resize, resize))

    if len(label.shape)==3:
        tensor = torch.from_numpy(label[:,:,0]).long().unsqueeze(0)
    else:
        tensor = torch.from_numpy(label).long().unsqueeze(0)
    size = tensor.size()

    oneHot_size = (size[0], 19, size[1], size[2])
    labels_real = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
    labels_real = labels_real.scatter_(1, tensor.view(size[0], 1, size[1], size[2]).data, 1.0)
    return labels_real

def bgr2yuv(bgr):
    """
    Convert the bgr channel to ycbcr. HWC 0-255
    :param bgr: bgr channel of the image. [16-235 ,16-240,16-240]
    :return: ycbcr channel.
    Y = 0.257*R' + 0.504*G' + 0.098*B' + 16
    U = -0.148*R' - 0.291*G' + 0.439*B' + 128
    V = 0.439*R' - 0.368*G' - 0.071*B' + 128
    """
    yuv = np.ones(bgr.shape).astype(np.float32)*16
    yuv[:, :, 0] = (0.256789 * bgr[:, :, 2] + 0.504129 * bgr[:, :, 1] + 0.097906 * bgr[:, :, 0]) + 16.0

    yuv[:, :, 1] = (-0.148223 * bgr[:, :, 2] - 0.290992 * bgr[:, :, 1] + 0.439215 * bgr[:, :, 0]) + 128.0

    yuv[:, :, 2] = (0.439215 * bgr[:, :, 2] - 0.367789 * bgr[:, :, 1] - 0.071426 * bgr[:, :, 0]) + 128.0

    return yuv
def rgb2yuv(rgb):
    """
    Convert the bgr channel to ycbcr. HWC
    :param bgr: bgr channel of the image.HWC 0-255
    :return: ycbcr channel. 0-255
    Y = 0.257*R' + 0.504*G' + 0.098*B' + 16
    U = -0.148*R' - 0.291*G' + 0.439*B' + 128
    V = 0.439*R' - 0.368*G' - 0.071*B' + 128
    """
    yuv = np.zeros(rgb.shape).astype(np.float32)*16
    yuv[:, :, 0] = (0.256789 * rgb[:, :, 0] + 0.504129 * rgb[:, :, 1] + 0.097906 * rgb[:, :, 2]) + 16

    yuv[:, :, 1] = (-0.148223 * rgb[:, :, 0] - 0.290992 * rgb[:, :, 1] + 0.439215 * rgb[:, :, 2]) + 128

    yuv[:, :, 2]= (0.439215 * rgb[:, :, 0] - 0.367789 * rgb[:, :, 1] - 0.071426 * rgb[:, :, 2]) + 128





    return yuv
def yuv2rgb(yuv):
    """
    Convert the ycbcr tensor channel to rgb tensor.
    :param y: y channel. BCHW
    :param cb: cb channel.
    :param cr: cr channel.
    :return: b, g, r channel. BCHW
    R = 1.164*Y + 1.596 * V - 222.9
    G = 1.164*Y - 0.392 * U - 0.823 * V+ 135.6
    B = 1.164*Y + 2.017 * U- 276.8
    """
    rgb = torch.FloatTensor(yuv.size()).zero_().to(yuv.get_device())
    for i in range(yuv.size()[0]):
        rgb[i,:,:,:]= _ycbcr2rgb(yuv[i].squeeze(0))  #CHW



    return rgb

def yuvnormal(yuv):
    """

    :param yuv:  16-235
    :return:      0-1
    """
    yuvnormal = np.zeros(yuv.shape).astype(np.float32)
    yuvnormal[:,:,0]=(yuv[:,:,0]-16.0)/219.0
    yuvnormal[:, :, 1] = (yuv[:, :, 1] - 16.0) / 224.0
    yuvnormal[:, :, 2] = (yuv[:, :, 2] - 16.0) / 224.0
    return yuvnormal

def yuvdenormal(yuv):
    """

    :param yuv: 0-1
    :return: 16-235
    """
    yuvdenormal = torch.FloatTensor(yuv.size()).zero_().to(yuv.get_device())
    for i in range(yuv.size()[0]):
        yuvdenormal[i, :, :, :] = _yuvdenormal(yuv[i].squeeze(0))  # CHW

    return yuvdenormal


def _yuvdenormal(yuv):
    """

    :param yuv: 0-1
    :return: 16-235
    """
    yuvdenormal = torch.FloatTensor(yuv.size()).zero_().to(yuv.get_device())
    yuvdenormal[:, :, 0] = yuv[:, :, 0] * 219 + 16
    yuvdenormal[:, :, 1] = yuv[:, :, 1] * 224 + 16
    yuvdenormal[:, :, 2] = yuv[:, :, 2] * 224 + 16
    return yuvdenormal




def _ycbcr2rgb(yuv):

    '''

    :param yuv: BCHW tensor 16-235>0-255 16-240>0-255
    :return: RGB BCHW tensor  0-1
    '''
    "r = np.floor(1.164383 * (y - 16.0) + 0 + 1.596027 * (cr - 128.0))"
    rgb = torch.FloatTensor(yuv.size()).zero_().to(yuv.get_device())
    r = (1.164383 * (yuv[0,:,:] - 16.0) + 0 + 1.596027 * (yuv[2,:,:] - 128.0 )).clamp(0.0,255.0)

    "g = np.floor(1.164383 * (y - 16.0) - 0.391762 * (cb - 128.0) - 0.812969 * (cr - 128.0))"
    g = (1.164383 * (yuv[0,:,:] - 16.0) - 0.391762 * (yuv[1,:,:] - 128.0) - 0.812969 * (yuv[2,:,:] - 128.0)).clamp(0.0,255.0)

    "b = np.floor(1.164383 * (y - 16.0) + 2.017230 * (cb - 128.0) + 0)"
    b = (1.164383 * (yuv[0,:,:] - 16.0) + 2.017230 * (yuv[1,:,:] - 128.0)).clamp(0.0,255.0)


    rgb[0, :, :] = r
    rgb[1, :, :] = g
    rgb[2, :, :] = b

    return rgb/255.0



def sep_iter_data(y, PAD):
    imagelist = []
    for p in range(0, y.shape[0], int(y.shape[0] / 2)+1):  # h
        for q in range(0, y.shape[1], int(y.shape[1] / 2)+1):  # w
            if p == 0 and q == 0:
                tmp = np.pad(y[p:p + int(y.shape[0] / 2+1 + PAD), q:q + int(y.shape[1] / 2+1 + PAD)],
                             ((PAD, 0), (PAD, 0), (0, 0)), mode='symmetric')
            elif p == 0 and q != 0:
                tmp = np.pad(y[p:p + int(y.shape[0] / 2+1 + PAD), q - PAD:], ((PAD, 0), (0, PAD), (0, 0)),
                             mode='symmetric')
            elif p != 0 and q == 0:
                tmp = np.pad(y[p - PAD:, q:q + int(y.shape[1] / 2+1 + PAD)], ((0, PAD), (PAD, 0), (0, 0)),
                             mode='symmetric')
            else:
                tmp = np.pad(y[p - PAD:, q - PAD:], ((0, PAD), (0, PAD), (0, 0)), mode='symmetric')
            imagelist.append(tmp)
            # print(p,q)
    return imagelist



def mergeimg(hrlist,sr_shape,PAD,SCALE,biash,biasw):
    merge_out = np.zeros([sr_shape[0], sr_shape[1], 3],np.uint8) #

    if PAD==0:
        merge_out[0: int(sr_shape[0] / 2+1), 0: int(sr_shape[1] / 2+1), :] = hrlist[0]
        merge_out[0: int(sr_shape[0] / 2+1), int(sr_shape[1] / 2+1):, :] =  hrlist[1]
        merge_out[int(sr_shape[0] / 2+1):, 0: int(sr_shape[1] / 2+1), :] = hrlist[2]
        merge_out[int(sr_shape[0] / 2+1):, int(sr_shape[1] / 2+1):, :] = hrlist[3]

    else:
        cnn_out_tmp = hrlist[0][PAD*SCALE:-PAD*SCALE, PAD*SCALE:-PAD*SCALE,:]
        merge_out[0: int(sr_shape[0] / 2+biash), 0: int(sr_shape[1] / 2+biasw),:] = cnn_out_tmp

        cnn_out_tmp = hrlist[1][PAD*SCALE:-PAD*SCALE, PAD*SCALE:-PAD*SCALE,:]
        merge_out[0: int(sr_shape[0] / 2+biash), int(sr_shape[1] / 2+biasw):,:] = cnn_out_tmp

        cnn_out_tmp = hrlist[2][PAD*SCALE:-PAD*SCALE, PAD*SCALE:-PAD*SCALE,:]
        merge_out[int(sr_shape[0] / 2+biash) :, 0: int(sr_shape[1] / 2+biasw),:] = cnn_out_tmp

        cnn_out_tmp = hrlist[3][PAD*SCALE:-PAD*SCALE, PAD*SCALE:-PAD*SCALE,:]
        merge_out[int(sr_shape[0] / 2+biash) :, int(sr_shape[1] / 2+biasw):,:] = cnn_out_tmp
    return merge_out
