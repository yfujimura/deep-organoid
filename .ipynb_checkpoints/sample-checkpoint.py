import os
import argparse
from collections import OrderedDict
import tqdm
import sys
import random
import numpy as np
import glob
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from loader import *
from model import *
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("checkpoint_dir")
    parser.add_argument("model", help="unet, resnet")
    parser.add_argument("--num_frame", type=int, default=1)
    parser.add_argument("--frame_interval", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--save_epoch", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lmd", type=float, default=0.005)
    parser.add_argument("--crop_width", type=int, default=512)
    parser.add_argument("--crop_height", type=int, default=512)
    parser.add_argument("--is_normalize", type=int, default=0)
    
    args = parser.parse_args()
    return args

            
            
def sample(net, test_loader, args):
    net.eval()
    running_loss = 0
    for data in tqdm.tqdm(test_loader):
        desc = data[0][2]
        batch_size, _, height, width = data[0][0].shape
        phase_img = torch.zeros(batch_size, 0, height, width)
        green_img = torch.zeros(batch_size, 0, height, width)
        for frame in range(args.num_frame):
            _phase_img = data[frame][0]
            _green_img = data[frame][1]
            phase_img = torch.cat((phase_img, _phase_img), 1)
            green_img = torch.cat((green_img, _green_img), 1)
        
        phase_img = phase_img.to("cuda")
        
        with torch.no_grad():
            pred = net(phase_img)
        
        input_img = phase_img[0].detach().cpu().numpy()
        gt = green_img[0,-1].detach().numpy()
        pred = pred[0,0].detach().to("cpu").numpy()
        
        gt = np.array([np.zeros_like(gt), gt, np.zeros_like(gt)]).transpose(1,2,0)
        gt = gt / 0.5
        gt[gt> 1] = 1
        pred = np.array([np.zeros_like(pred), pred, np.zeros_like(pred)]).transpose(1,2,0)
        pred = pred / 0.5
        pred[pred> 1] = 1
        
        
        cell_num = desc[0].item()
        day = desc[1].item()
        z_stack = desc[2].item()
        
        Image.fromarray((input_img[0]*255).astype(np.uint8)).save("results/sample/img/{}_{}_{}.png".format(cell_num, day, z_stack))
        Image.fromarray((gt*255).astype(np.uint8)).save("results/sample/gt/{}_{}_{}.png".format(cell_num, day, z_stack))
        Image.fromarray((pred*255).astype(np.uint8)).save("results/sample/pred/{}_{}_{}.png".format(cell_num, day, z_stack))
        
            
        
        

def main():
    args = parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    test_list = [64]
    
    if args.is_normalize == 1:
        is_normalize = True
    else:
        is_normalize = False
        
    test_dataset = CenterCropDataset("dataset/sample", test_list, "center_pixels", 
                                     is_random_flip=False, return_desc=True, 
                                     num_frame=args.num_frame, frame_interval=args.frame_interval,
                                     is_normalize=is_normalize, 
                                     crop_width=args.crop_width, crop_height=args.crop_height, 
                                    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    
    net = get_model(args)
    net.to("cuda")
    
    ckpt_dir = "{}/{}_{}".format(args.checkpoint_dir, args.lmd, args.gamma)
    ckpt = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    import_params(net, ckpt[0])
    
    create_dir("results")
    create_dir("results/sample")
    create_dir("results/sample/img")
    create_dir("results/sample/gt")
    create_dir("results/sample/pred")
    sample(net, test_loader, args)

if __name__ == "__main__":
    main()