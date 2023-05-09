import os
import argparse
from collections import OrderedDict
import tqdm
import sys
import random
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

def torch_fix_seed(seed=0):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 0
torch_fix_seed(seed)

from loader import *
from model import *
from utils import *
from loss import *

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("checkpoint_dir")
    parser.add_argument("model", help="unet, resnet")
    parser.add_argument("--num_frame", type=int, default=1)
    parser.add_argument("--frame_interval", type=int, default=0)
    parser.add_argument("--epoch0", type=int, default=0)
    parser.add_argument("--epoch_num", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--save_epoch", type=int, default=1)
    parser.add_argument("--loss_fn", type=int, default=4, help="0: MSE, 1:, L1, 2: focal loss, 3: cos similarity, 4: MSE and cos similarity, 5: adaptive MSE and cossim")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lmd", type=float, default=0.005)
    parser.add_argument("--crop_width", type=int, default=512)
    parser.add_argument("--crop_height", type=int, default=512)
    parser.add_argument("--is_normalize", type=int, default=0)
    parser.add_argument("--lr_decay", type=int, default=1)
    parser.add_argument("--decay_rate", type=float, default=0.1)
    parser.add_argument("--optimizer", default="ADAM", help="ADAM or SGD or RADAM or ADAMW")
    
    args = parser.parse_args()
    return args


def train(net, train_loader, test_loader, args):
    checkpoint_dir = args.checkpoint_dir
    epoch_num = args.epoch_num
    save_epoch = args.save_epoch
    epoch0 = args.epoch0
    lr = args.lr
    
    params = net.parameters()
    
    if args.loss_fn == 0:
        loss_fn = nn.MSELoss()
    elif args.loss_fn == 1:
        loss_fn = nn.L1Loss()
    elif args.loss_fn == 2:
        loss_fn = FocalLoss(gamma=args.gamma)
    elif args.loss_fn == 3:
        loss_fn = CosineSimilarityLoss()
    elif args.loss_fn == 4:
        loss_fn = MSEandCosineSimilarityLoss(gamma=args.gamma, lmd=args.lmd)
    elif args.loss_fn == 5:
        loss_fn = AdaptiveMSEandCosineSimilarityLoss()
        loss_fn.to("cuda")
        params = list(params) + list(loss_fn.parameters())
    
    
    train_losses = []
    val_losses = []
    if args.optimizer == "ADAM":
        optimizer_cls=optim.Adam
        optimizer = optimizer_cls(params, lr=lr)
    elif args.optimizer == "SGD":
        optimizer_cls = optim.SGD
        optimizer = optimizer_cls(params, lr=lr, momentum=0.9, weight_decay=0.0005,)
    elif args.optimizer == "RADAM":
        optimizer_cls = optim.RAdam
        optimizer = optimizer_cls(params, lr=lr)
    elif args.optimizer == "ADAMW":
        optimizer_cls=optim.AdamW
        optimizer = optimizer_cls(params, lr=lr)
        
    if args.lr_decay == 1:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80], gamma=args.decay_rate)
    
    if epoch0 != 0:
        import_params(net, os.path.join(checkpoint_dir, "epoch_{:04}.pth".format(epoch0)))
        import_params(optimizer, os.path.join(checkpoint_dir, "optimizer_epoch_{:04}.pth".format(epoch0)))
        train_losses = np.load(os.path.join(checkpoint_dir, "train_loss_epoch_{:04}.npy".format(epoch0))).tolist()
        val_losses = np.load(os.path.join(checkpoint_dir, "test_loss_epoch_{:04}.npy".format(epoch0))).tolist()
        
    
    for epoch in range(epoch0, epoch_num):
        running_loss = 0
        net.train()
        
        with tqdm.tqdm(train_loader) as pbar:
            for i, data in enumerate(pbar):
                batch_size, _, height, width = data[0][0].shape
                phase_img = torch.zeros(batch_size, 0, height, width)
                green_img = torch.zeros(batch_size, 0, height, width)
                for frame in range(args.num_frame):
                    _phase_img = data[frame][0]
                    _green_img = data[frame][1]
                    phase_img = torch.cat((phase_img, _phase_img), 1)
                    green_img = torch.cat((green_img, _green_img), 1)
                    
                
                #phase_img = phase_img.unsqueeze(1).to("cuda")
                #green_img = green_img.unsqueeze(1).to("cuda")
                
                phase_img = phase_img.to("cuda")
                green_img = green_img.to("cuda")
                
                pred = net(phase_img)
                if pred.shape[2] != green_img.shape[2]:
                    green_img = F.interpolate(green_img, size=(pred.shape[2], pred.shape[3]), mode="bilinear", align_corners=True)
                
                
                loss = loss_fn(pred, green_img[:,-1,:,:].unsqueeze(1))
                running_loss = (running_loss * i + loss.item()) / (i + 1)
                pbar.set_postfix(OrderedDict(Loss=running_loss))
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                
               
               
        train_losses.append(running_loss)
        val_losses.append(test(net, test_loader, args))
        print(epoch, "train_loss: {}".format(train_losses[-1]), "test_loss: {}".format(val_losses[-1]))
        
        if (epoch+1) % save_epoch == 0: 
            fn = os.path.join(checkpoint_dir, 'epoch_{:04}.pth'.format(epoch+1))
            params = net.state_dict()
            torch.save(params, fn, pickle_protocol=4)
            fn = os.path.join(checkpoint_dir, 'optimizer_epoch_{:04}.pth'.format(epoch+1))
            torch.save(optimizer.state_dict(), fn)
            
            np.save(os.path.join(checkpoint_dir, 'train_loss_epoch_{:04}'.format(epoch+1)), np.array(train_losses))
            np.save(os.path.join(checkpoint_dir, 'test_loss_epoch_{:04}'.format(epoch+1)), np.array(val_losses))
            
            if args.loss_fn == 5:
                fn = os.path.join(checkpoint_dir, 'weight_epoch_{:04}.pth'.format(epoch+1))
                params = loss_fn.state_dict()
                torch.save(params, fn, pickle_protocol=4)
                
        if args.lr_decay == 1:
            scheduler.step()
            
            
            
def test(net, test_loader, args, loss_fn=nn.MSELoss()):
    net.eval()
    running_loss = 0
    for data in tqdm.tqdm(test_loader):
        batch_size, _, height, width = data[0][0].shape
        phase_img = torch.zeros(batch_size, 0, height, width)
        green_img = torch.zeros(batch_size, 0, height, width)
        for frame in range(args.num_frame):
            _phase_img = data[frame][0]
            _green_img = data[frame][1]
            phase_img = torch.cat((phase_img, _phase_img), 1)
            green_img = torch.cat((green_img, _green_img), 1)
        
        #phase_img = phase_img.unsqueeze(1).to("cuda")
        #green_img = green_img.unsqueeze(1).to("cuda")
        
        phase_img = phase_img.to("cuda")
        green_img = green_img.to("cuda")
        
        with torch.no_grad():
            pred = net(phase_img)
            if pred.shape[2] != green_img.shape[2]:
                green_img = F.interpolate(green_img, size=(pred.shape[2], pred.shape[3]), mode="bilinear", align_corners=True)
            loss = loss_fn(pred, green_img[:,-1,:,:].unsqueeze(1))
            running_loss += loss.item()
        
    val_loss = running_loss / len(test_loader)
    return val_loss
        
        

def main():
    args = parse_args()
    
    create_dir(args.checkpoint_dir)
    write_config(args)
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    test_list = []
    train_list = []
    for i in range(1, 97):
        if i % 8 == 0:
            test_list.append(i)
        else:
            train_list.append(i)
    print("train cells:", train_list)
    print("validation cells:", test_list[:len(test_list)//2])
    print("test cells:", test_list[len(test_list)//2:])
    
    if args.is_normalize == 1:
        is_normalize = True
    else:
        is_normalize = False
        
    test_dataset = CenterCropDataset("../dataset/organoid_211220", test_list[:len(test_list)//2], "../center_pixels", 
                                     is_random_flip=False, return_desc=False, 
                                     num_frame=args.num_frame, frame_interval=args.frame_interval,
                                     is_normalize=is_normalize, 
                                     crop_width=args.crop_width, crop_height=args.crop_height, 
                                    )
    
    train_dataset = CenterCropDataset("../dataset/organoid_211220", train_list, "../center_pixels", 
                                     is_random_flip=True, return_desc=False, 
                                     num_frame=args.num_frame, frame_interval=args.frame_interval,
                                     is_normalize=is_normalize, 
                                     crop_width=args.crop_width, crop_height=args.crop_height, 
                                    )
    
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    net = get_model(args)
    net.to("cuda")
    
    train(net, train_loader, test_loader, args)

if __name__ == "__main__":
    main()