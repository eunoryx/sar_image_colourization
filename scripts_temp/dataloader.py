import torch
from torch.utils.data import Dataset, DataLoader

import cv2 as cv
import numpy as np
import os
from PIL import Image

class SAR_set(Dataset):                                                         #dataset class

    def __init__(self):                                                         #directory/path stuff

        self.img_dir = "data"   
        self.folder_path = sorted(os.listdir(self.img_dir))
        self.colour_path = os.path.join(self.img_dir, self.folder_path[0])
        self.gray_path = os.path.join(self.img_dir, self.folder_path[1])
        self.len_data = len(os.listdir(self.gray_path))

    def path_to_np(self, filepath, grayscale):                                  #ndarray convert
        img = Image.open(filepath)

        if grayscale:
            img = img.convert("L")

        img = img.resize((128, 128))
        img_np = np.array(img)

        return img_np

    def rgb_to_lab(self, np_arr):                                               #rgb to lab tensor conversion

        img_lab = cv.cvtColor(np_arr, cv.COLOR_RGB2LAB)
        img_lab_tensor = torch.tensor(img_lab)

        return img_lab_tensor

    def __len__(self):                                                          #len fn

        return self.len_data
    
    def __getitem__(self,idx):                                                  #getitem; returns L info as well as LAB info 

        colour_list = os.listdir(self.colour_path)
        colour_file_path = os.path.join(self.colour_path, colour_list[idx])
        img_np = self.path_to_np(colour_file_path, 0)
        img_lab_tensor = self.rgb_to_lab(img_np)

        gray_list = os.listdir(self.gray_path)
        gray_file_path = os.path.join(self.gray_path, gray_list[idx])
        gray_tensor = self.path_to_np(gray_file_path, 1)
        gray_tensor = torch.tensor(gray_tensor)

        return gray_tensor, img_lab_tensor

dataset = SAR_set()                                                                                 #dataset

train_size = int(0.8 * len(dataset))                                                                #test-train split
test_size = len(dataset) - train_size  
train_dataset,test_dataset = (dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False)                           #test-train dataloaders
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
