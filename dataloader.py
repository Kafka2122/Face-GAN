import os 
import cv2
import random
import numpy as np
class Custom_Dataloader():
    def __init__(self, dir_path = None, mode = None):
        
        if mode == "train":
            self.dir_path = dir_path + "/celeba_hq/train/male"
        else:
            self.dir_path = dir_path + "/celeba_hq/val/male"

        self.image_files = os.listdir(self.dir_path)
        

    def __len__(self): return len(self.image_files)

    def pair_image(self, image_file):

        img_path = os.path.join(self.dir_path, image_file)
        img1 = cv2.imread(img_path).astype(np.float32)/ 255.0
        img1 = cv2.resize(img1, (128, 128))  # Resize image to shape 128x128

        random_image_file = random.choice(self.image_files)
        random_img_path = os.path.join(self.dir_path, random_image_file)
        img2 = cv2.imread(random_img_path).astype(np.float32)/ 255.0
        img2 = cv2.resize(img2, (128, 128))  # Resize image to shape 128x128

        image_pair_dict = {"img1": img1, "img2": img2}
        return image_pair_dict

    def __getitem__(self, index):
        
        image = self.image_files[index]
        pair_images = self.pair_image(image)
        return pair_images
    