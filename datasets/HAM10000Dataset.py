import os
import sys
import torch
from utils import general
import cv2
import h5py
import json
import collections
import numpy as np
from os.path import join as oj
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from PIL import Image
import time
from tqdm import tqdm

import datasets.common as common
import albumentations as albu










@common.setdatasetname("HAM10000Dataset")
class HAM10000Dataset(BaseDataset):
    def __init__(self, mode, args):
        self.mode = mode
        self.seed = args.seed
        np.random.seed(self.seed)
        self.fold = args.trial
        self.mask_size = args.mask_data_size
        self.mask_ratio = args.mask_ratio if self.mode == "train" else 1.0 # 전체 mask중에 이 만큼만 mask 사용한다는 뜻
        self.data_path = args.data_path
        self.patch_factory = common.Patch(224, args.patch_transparency)
        assert args.what_to_patch_on_train in ['benign', 'malignant', 'all', None]
        assert args.what_to_patch_on_val in ['benign', 'malignant', 'all', None]

        total_datanum = None
        label = None
        with h5py.File(os.path.join(self.data_path, 'dataset_in_use.hdf5'), 'r') as f:
            total_datanum = f['img'].shape[0]
            label = f['label'][()]

        idx_list = np.arange(total_datanum)
        benign_idx = []
        malignant_idx = []
        for idx in idx_list:
            if label[idx] == 0:
                benign_idx.append(idx)
            elif label[idx] == 1:
                malignant_idx.append(idx)
            else:
                raise Exception('')
        
        # mode가 train인지 val인지를 고려해 사용할 data의 idx를 benign, malignant로 분류함
        benign_train_idx, benign_valid_idx = self.get_idx_to_use(benign_idx)
        malignant_train_idx, malignant_valid_idx = self.get_idx_to_use(malignant_idx)

        # train data만 이용해서 mean, std 구하기
        train_idx = np.append(benign_train_idx, malignant_train_idx)
        print('train_idx :', train_idx)
        mean, std = self.get_mean_std(train_idx)
        print('mean : {}\nstd : {}'.format(list(mean), list(std)))

        # benign과 malignant를 합쳐서 하나의 train or val dataset의 idx array를 만든다
        main_idx = None
        if self.mode == "train":
            main_idx = np.append(benign_train_idx, malignant_train_idx)
        elif self.mode == "valid":
            main_idx = np.append(benign_valid_idx, malignant_valid_idx)
        else:
            raise Exception("Unknown mode : {}".format(self.mode))


        self.img = []
        self.seg = []
        self.label = []
        # np.random.shuffle(main_idx)이 필요한 이유
        # 1. 뒤에 small_dataset을 만들때 제일 앞에 위치한 일부 data만 사용되기 때문에 섞어줘야 함
        # 2. mask ratio 실험할 때 mask들 중 앞에 위치한 일부 mask들만 사용되기 때문에 섞어줘야 함
        np.random.shuffle(main_idx)
        with h5py.File(os.path.join(self.data_path, 'dataset_in_use.hdf5'), 'r') as f:
            for idx in main_idx:
                if self.mode == 'train':
                    if args.what_to_patch_on_train == 'all':
                        new_img = self.patch_factory.put_patch(f['img'][idx])
                    elif args.what_to_patch_on_train == 'benign':
                        if f['label'][idx] == 0:
                            new_img = self.patch_factory.put_patch(f['img'][idx])
                        else:
                            new_img = f['img'][idx]
                    elif args.what_to_patch_on_train == 'malignant':
                        if f['label'][idx] == 0:
                            new_img = f['img'][idx]
                        else:
                            new_img = self.patch_factory.put_patch(f['img'][idx])
                    else:
                        new_img = f['img'][idx]
                else:
                    if args.what_to_patch_on_val == 'all':
                        new_img = self.patch_factory.put_patch(f['img'][idx])
                    elif args.what_to_patch_on_val == 'benign':
                        if f['label'][idx] == 0:
                            new_img = self.patch_factory.put_patch(f['img'][idx])
                        else:
                            new_img = f['img'][idx]
                    elif args.what_to_patch_on_val == 'malignant':
                        if f['label'][idx] == 0:
                            new_img = f['img'][idx]
                        else:
                            new_img = self.patch_factory.put_patch(f['img'][idx])
                    else:
                        new_img = f['img'][idx]
                self.img.append((new_img/255.0-mean)/std) # broadcasting 잘 되는 것 확인함
                self.seg.append(f['seg'][idx])
                self.label.append(f['label'][idx])

        # print('main_idx.shape :', main_idx.shape)
        # print('main_idx.shape[0] :', main_idx.shape[0])
        # print('self.mask_ratio :', self.mask_ratio)
        boundary = int(main_idx.shape[0] * self.mask_ratio)
        # print('boundary :', boundary)
        self.use_mask = np.zeros_like(main_idx)
        self.use_mask[:boundary] = 1

        if args.small_dataset:
            self.img = self.img[:50]
            self.seg = self.seg[:50]
            self.label = self.label[:50]
            self.use_mask = self.use_mask[:50]

        self.augmentation = common.data_augmentation()
        self.resize = albu.Resize(self.mask_size, self.mask_size)
        self.preprocessing = common.data_preprocessing()

             
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        img = self.img[index]
        seg = self.seg[index]
        label = self.label[index]
        use_mask = self.use_mask[index]
        
        if self.mode == "train":
            sample = self.augmentation(image=img, mask=seg)
            img, seg = sample['image'], sample['mask']
            seg = self.resize(image=seg)['image']

        elif self.mode == 'valid' and self.mask_size is not None:
            seg = self.resize(image=seg)['image']
        sample = self.preprocessing(image=img, mask=seg)
        img, seg = sample['image'], sample['mask']
        
        return (img, seg), label, use_mask
        
        
    def get_idx_to_use(self, idx_array):
        idx_list = list(idx_array)
        np.random.shuffle(idx_list)

        fold_num = 5
        fold_size = len(idx_list) / fold_num
        split_list = [int(i*fold_size) for i in range(fold_num+1)]

        train_idx = idx_list[:split_list[self.fold]] + idx_list[split_list[self.fold+1]:]
        valid_idx = idx_list[split_list[self.fold] : split_list[self.fold+1]]
        return np.array(train_idx), np.array(valid_idx)

    def get_mean_std(self, idx_array):
        print('getting mean, std...')
        img = []
        with h5py.File(os.path.join(self.data_path, 'dataset_in_use.hdf5'), 'r') as f:
            for idx in idx_array:
                if f['label'][idx] == 0:
                    new_img = self.patch_factory.put_patch(f['img'][idx])
                else:
                    new_img = f['img'][idx]
                img.append(new_img/255.0)

        img = np.stack(img, axis=0)
        mean = np.mean(img, (0, 1, 2))
        std = np.std(img, (0, 1, 2))
        return mean, std
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    
    
    
    
    
    
    
    