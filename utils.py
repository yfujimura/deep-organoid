import os
import numpy as np
from PIL import Image
import torch
import datetime

from model import *

def get_model(args):
    if args.model == "unet":
        model = unet.Unet(num_frame=args.num_frame)
    elif args.model == "resnet":
        backbone = resnet.resnet50(pretrained=True)
        encoder = aenet.Encoder(backbone)
        model = aenet.AENet(encoder)
    return model
    

def read_image(filename):
    img = np.array(Image.open(filename)).astype(np.float32) / 255.
    return img

def crop_image(img, crop_size):
    # crop_size = (crop_width, crop_height)
    shape = img.shape
    height = shape[0]
    width = shape[1]
    
    center_y = height // 2
    center_x = width // 2
    y0 = center_y - crop_size[1] // 2
    x0 = center_x - crop_size[0] // 2
    
    cropped = img[y0:y0+crop_size[1], x0:x0+crop_size[0]].copy()
    return cropped

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
def import_params(model, filename):
    params = torch.load(filename)
    model.load_state_dict(params)
    
def write_config(args):
    now = datetime.datetime.now()
    if args.epoch0 == 0:
        with open(os.path.join(args.checkpoint_dir, "config.txt"), "w") as f:
            f.write(now.strftime('%Y/%m/%d %H:%M:%S\n'))
            _write_config(args, f)
    else:
        with open(os.path.join(args.checkpoint_dir, "config.txt"), "a") as f:
            f.write(now.strftime('\n%Y/%m/%d %H:%M:%S\n'))
            _write_config(args, f)
            
def _write_config(args, f):
    f.write("checkpoint dir: {}\n".format(args.checkpoint_dir))
    f.write("model: {}\n".format(args.model))
    f.write("num frame: {}\n".format(args.num_frame))
    f.write("frame interval: {}\n".format(args.frame_interval))
    f.write("batch size: {}\n".format(args.batch_size))
    f.write("epoch num: {}\n".format(args.epoch_num))
    f.write("epoch0: {}\n".format(args.epoch0))
    if args.loss_fn == 0:
        f.write("loss fn: MSE\n")
    elif args.loss_fn == 1:
        f.write("loss fn: L1\n")
    elif args.loss_fn == 2:
        f.write("loss fn: focal loss, gamma={}\n".format(args.gamma))
    elif args.loss_fn == 3:
        f.write("loss fn: cos similarity\n")
    elif args.loss_fn == 4:
        f.write("loss fn: MSE and cos similarity, lmd={}, gamma={}\n".format(args.lmd,args.gamma))
    elif args.loss_fn == 5:
        f.write("loss fn: adaptive MSE and cos similarity")
    f.write("learning rate: {}\n".format(args.lr))
    f.write("crop width: {}\n".format(args.crop_width))
    f.write("crop height: {}\n".format(args.crop_height))
    if args.is_normalize == 1:
        f.write("normalize: True\n")
    else:
        f.write("normalize: False\n")
    f.write("optimzier: {}\n".format(args.optimizer))
    if args.lr_decay == 1:
        f.write("lr decay: True\n")
        f.write("decay rate: {}\n".format(args.decay_rate))
    else:
        f.write("lr decay: False\n")
    
    
    