import os
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

from utils import *
    
class CenterCropDataset(Dataset):
    
    def __init__(self, root_dir, cell_list, center_pixels_dir, z_num=14, is_random_flip=False, return_desc=False, num_frame=1, frame_interval=0, 
                 width=960, height=720, crop_width=512, crop_height=512,
                 is_normalize=False):
        
        self.root_dir = root_dir
        self.cell_list = cell_list
        self.center_pixels_dir = center_pixels_dir
        self.is_random_flip = is_random_flip
        self.num_frame = num_frame
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.is_normalize = is_normalize
        self.width = width
        self.height = height
        
        self.days = [4.5,5,6,7,8]
        self.z_num = z_num
        self.channels = [1, 4]
        
        self.dataset = []
        
        for c in self.cell_list:
            for d in range(len(self.days) - (num_frame-1)*frame_interval):
                for z in range(self.z_num):
                    data = []
                    for i in range(num_frame):
                        data.append([c, self.days[d+i*frame_interval],z+1])
                    self.dataset.append(data)
                    
                    
        self.return_desc = return_desc
        
    def __len__(self):
        return len(self.dataset)
    
    def transform(self, img1, img2, cy, cx):
        # resize and crop
        
        y0 = int(cy - self.crop_height/2)
        x0 = int(cx - self.crop_width/2)
        img1 = TF.crop(img1, y0, x0, self.crop_height, self.crop_width)
        img2 = TF.crop(img2, y0, x0, self.crop_height, self.crop_width)
        
        if self.is_random_flip:
            # random horizontal flipping
            if random.random() > 0.5:
                img1 = TF.hflip(img1)
                img2 = TF.hflip(img2)
                
            # random vertical flipping
            if random.random() > 0.5:
                img1 = TF.vflip(img1)
                img2 = TF.vflip(img2)

        # transform to tensor
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)
            
        return img1, img2
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        out = []
        for frame in range(self.num_frame):
            cell_num = data[frame][0]
            day = data[frame][1]
            z_num = data[frame][2]
            
            filename = os.path.join(self.root_dir, 
                                    "Day{}".format(day), 
                                    "XY{:02}".format(cell_num), 
                                    "Day{}_XY{:02}_Z{:03}_CH{}.tif".format(day, cell_num, z_num, self.channels[1]))
            phase_img = Image.open(filename)
            
            filename = os.path.join(self.root_dir, 
                                    "Day{}".format(day), 
                                    "XY{:02}".format(cell_num), 
                                    "Day{}_XY{:02}_Z{:03}_CH{}.tif".format(day, cell_num, z_num, self.channels[0]))
            green_img = Image.open(filename)
            
            cy, cx = np.loadtxt(os.path.join(self.center_pixels_dir,
                                            "Day{}_XY{:02}_Z{:03}_CH4.txt".format(day, cell_num, z_num)))
            if cy + self.crop_height/2 > self.height:
                cy = self.height - self.crop_height//2
            if cx + self.crop_width/2 > self.width:
                cx = self.width - self.crop_width//2
            
            phase_img, green_img = self.transform(phase_img, green_img, cy, cx)
            
            if self.is_normalize:
                min_value = torch.min(green_img)
                max_value = torch.max(green_img)
                green_img = (green_img - min_value) / (max_value - min_value)
                
            
            green_img = green_img[1,:,:].unsqueeze(0)
                
            if self.return_desc:
                out.append([phase_img, green_img, [cell_num, day, z_num]])
            else:
                out.append([phase_img, green_img])
        return out


        
        
        