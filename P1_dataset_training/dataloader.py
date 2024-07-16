import os 
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class Custom_loader(Dataset):
    def __init__(self,data_folder):
        self.data_folder = data_folder
        self.files = sorted(os.listdir(data_folder))
        
    def __len__(self): return len(self.files)
        
    def pair_image(self,file_name):
        img_pair = {}
        random_img_list = [img for img in self.files if img  != file_name ] # select the images except the first img
        random_img = random.choice(random_img_list)

        img1 = cv2.imread(file_name).astype(np.float32)/255.0
        img1 = cv2.resize(img1,(128,128))
        img_pair['img1'] = img1
       
        pic_img2 = random.choice(os.listdir(os.path.join(self.data_folder, random_img)))

        img2 = cv2.imread(os.path.join(self.data_folder, random_img,pic_img2)).astype(np.float32)/255.0
        img2 = cv2.resize(img2,(128,128))
        img_pair['img2'] = img2
        
        return img_pair
    
    def __getitem__(self,idx):
         # create a list of all the folder classes
        name = self.files[idx] # select specific folder or class
        pic = random.choice(os.listdir(os.path.join(self.data_folder, name))) # select an image from the identity folder
        
        file_name = os.path.join(self.data_folder, name,pic)

        data = self.pair_image(file_name)
        
        return data
