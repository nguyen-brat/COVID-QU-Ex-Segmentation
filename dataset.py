import os
import cv2
import torch
import numpy as np
import imgaug.augmenters as iaa
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class Covid(Dataset):
    def __init__(self, rootpath, cropsize=(256, 256), mode='train', *args, **kwargs):
        super(Covid, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.cropsize=cropsize

        self.rootpth = rootpath
        self.imgs = []

        class_dir = os.listdir(os.path.join(self.rootpth,self.mode.capitalize()))  # class
        self.class_cvt = { 'Normal':0, 'COVID-19':1,'Non-COVID':2}
    
        for _class in class_dir:
            self.class_path = os.path.join(self.rootpth,self.mode.capitalize())
            self.img_path = os.path.join(self.class_path,_class,'images')
            img_lst = os.listdir(self.img_path)
            img_lst.sort()
            for img in img_lst:
                self.imgs.append((img,self.class_cvt[_class]))
        #  pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_color = iaa.Sequential([
            iaa.LinearContrast((0.4, 1.6)),
            ])
        self.trans_train = iaa.Sequential([
            iaa.Fliplr(0.5),
            ])

    def __getitem__(self, idx):
        impth = self.imgs[idx]
        self.img_path = os.path.join(self.class_path,list(self.class_cvt.keys())[impth[1]],'images')
        
        img =  cv2.imread(os.path.join(self.img_path,impth[0]))
        img = cv2.resize(img,self.cropsize, cv2.INTER_LINEAR)
        label = F.one_hot(torch.tensor(impth[1]), num_classes=3)
        
        mask_path = self.img_path.replace('images','lung masks')
        lung_mask = cv2.imread(os.path.join(mask_path,impth[0]),0)
        lung_mask = cv2.resize(lung_mask,self.cropsize, cv2.INTER_NEAREST ).astype(np.int64)        
        
        infect_path = self.img_path.replace('images','infection masks')
        infect_mask = cv2.imread(os.path.join(infect_path,impth[0]),0)
        infect_mask = cv2.resize(infect_mask,self.cropsize, cv2.INTER_NEAREST ).astype(np.int64)
      
        if self.mode == 'train':
            det_tf = self.trans_train.to_deterministic()
            img = det_tf.augment_image(img)
            lung_mask = det_tf.augment_image( lung_mask)
            infect_mask = det_tf.augment_image(infect_mask)

        img = self.to_tensor(img)        
        lung_mask = np.where(lung_mask !=0,1.,0.)
        lung_mask = torch.from_numpy(lung_mask.astype(np.float32)).clone()
        lung_mask = F.one_hot(lung_mask.long(), num_classes=2)
        lung_mask = lung_mask.to(torch.float32)
        lung_mask = lung_mask.permute(2, 0, 1)
        
        infect_mask = np.where(infect_mask !=0,1.,0.)
        infect_mask = torch.from_numpy(infect_mask.astype(np.float32)).clone()
        infect_mask = F.one_hot(infect_mask.long(), num_classes=2)
        infect_mask = infect_mask.to(torch.float32)
        infect_mask = infect_mask.permute(2, 0, 1)
        
        return img, label, lung_mask, infect_mask

    def __len__(self):
        return len(self.imgs)