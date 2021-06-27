import os
import re
import cv2
import h5py
import torch
import seaborn
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect

import numpy as np
from PIL import Image
from PIL import Image, ImageDraw


def extract_digit(my_str):
    return int(''.join(list(filter(str.isdigit, my_str))))

def extract_str(my_str):
    return ''.join([char for char in my_str if not char.isdigit()])

def save_hdf5(path, key, data, no_maxshape=False, compression=False):
    with h5py.File(path, 'a') as f:
        if key not in f: 
            if no_maxshape:
                f.create_dataset(key, data=data, compression=compression) # 압축하고 싶으면 compression='gzip'
            else:
                maxshape = tuple(None for i in range(len(data.shape)))
                f.create_dataset(key, data=data, maxshape=maxshape)
        else:
            f[key].resize((f[key].shape[0] + data.shape[0]), axis=0)
            f[key][-data.shape[0]:] = data


def get_class_attr(Cls):
    return [a for a, v in Cls.__dict__.items()
              if not re.match('<function.*?>', str(v))
              and not (a.startswith('__') and a.endswith('__'))]

def get_class_attr_val(cls):
    attr = get_class_attr(cls)
    attr_dict = {}
    for a in attr:
        attr_dict[a] = getattr(cls, a)
    return attr_dict


def save_args(path, args):
    if not os.path.exists(path):
        os.makedirs(path)
        
    with open(path + '/args.txt', 'w') as f:
        attr_val = get_class_attr_val(args)
        for k, v in attr_val.items():
            f.write(str(k) + " = " + str(v) + "\n")
            
            
def imshow_depth(img, start_depth=0, showing_img_num=None, figsize=(20, 20)):
    depth = img.shape[2]
    if showing_img_num == None:
        fig, ax = plt.subplots(nrows=1, ncols=depth, figsize=figsize)
        for depth_num in range(depth):
            ax[depth_num].imshow(img[:,:,start_depth+depth_num], cmap='gray')
            ax[depth_num].axis('off')
        plt.show()
    else:
        fig, ax = plt.subplots(nrows=1, ncols=showing_img_num, figsize=(20, 20))
        for depth_num in range(showing_img_num):
            ax[depth_num].imshow(img[:,:,start_depth+depth_num], cmap='gray')
            ax[depth_num].axis('off')
        plt.show()
        
        
def imshow_depth_seaborn(img, start_depth=0, showing_img_num=None, cbar=True, figsize=(20, 20)):
    depth = img.shape[2]
    if showing_img_num == None:
        fig, ax = plt.subplots(nrows=1, ncols=depth, figsize=figsize)
        for depth_num in range(depth):
            seaborn.heatmap(img[:,:,start_depth+depth_num], cmap='RdBu_r', center=0, ax=ax[depth_num], square=True, cbar_kws={'shrink': .2})
        plt.axis('off')
        plt.show()
    else:
        fig, ax = plt.subplots(nrows=1, ncols=showing_img_num, figsize=(20, 20))
        for depth_num in range(showing_img_num):
            seaborn.heatmap(img[:,:,start_depth+depth_num], cmap='RdBu_r', center=0, ax=ax[depth_num], square=True, cbar=cbar, cbar_kws={'shrink': .2})
        plt.axis('off')
        plt.show()
        
        

           
def hist(array, start=None, end=None, unit=None, figsize=(10, 5)):
    if start == None:
        start = np.min(array)
    if end == None:
        end = np.max(array)
    if unit == None:
        unit = (end - start) / 10
    
    print('start : {}, end : {}, unit: {}'.format(start, end, unit))
    
    x = [start+unit*i for i in range(1000) if start+unit*i < end]
    x += [x[-1] + unit]
    y = [0 for i in range(len(x))]
    print('x :', x)
    print('y :', y)
    for i in range(array.shape[0]):
#         print(i)
        quo = int((array[i]-start) // unit)
        if 0 <= quo and quo <= len(y):
#             print(quo, array[i])
            y[quo] += 1
        
    plt.figure(figsize=figsize)
    plt.bar(np.arange(len(y)), y, tick_label=[round(i, 1) for i in x], align='center')

    plt.show()
            

def resize(img, shrink_size):
    # cv2.resize는 shape length가 2인 matrix만 resize할 수 있어서 한장한장 resize한 다음에 다 붙일거임
    other_dim = img.shape[:-2]
    if len(img.shape) > 2:
        img = img.reshape(np.prod(img.shape[:-2]), *img.shape[-2:])
    
    img_resized = []
    for num in range(img.shape[0]):
        one_resized_img = cv2.resize(img[num], shrink_size, interpolation = cv2.INTER_AREA)
        img_resized.append(np.expand_dims(one_resized_img, axis=0))

    img_resized = np.concatenate(img_resized, axis=0)
    img_resized = img_resized.reshape(*other_dim, *img_resized.shape[-2:])
    
    return img_resized


def get_setting(path):
    setting_dict = collections.OrderedDict()
    with open(os.path.join(path, 'args.txt'), 'r') as f:
        for line in f: 
            key, val = line.split(' = ')
            setting_dict[key] = [val[:-1]]
    return setting_dict

def get_setting2(path):
    setting_dict = collections.OrderedDict()
    with open(os.path.join(path, 'args.txt'), 'r') as f:
        for line in f: 
            key, val = line.split(' : ')
            setting_dict[key] = [val[:-1]]
    return setting_dict

def write_setting(setting_dict, path):
    with open(os.path.join(path, 'args.txt'), 'w') as f:
        for key in list(setting_dict.keys()):
            f.write(key+' = '+setting_dict[key][0]+'\n')
        

def result2dataframe(hyper_paths, filter='20'):
    frame_dict = collections.OrderedDict()
#     hyper_paths = [os.path.join(path, dir) for dir in os.listdir(path) if dir[:len(filter)] == filter]
    for hyper_path in hyper_paths:
        setting_dict = get_setting(hyper_path)

        num_of_previous_values = 0
        if len(frame_dict) == 0:
            frame_dict = setting_dict
        else:
            num_of_previous_values = len(frame_dict[list(frame_dict.keys())[0]])
            for key in setting_dict:
                if key not in frame_dict:
                    frame_dict[key] = ['-' for i in range(num_of_previous_values)]
                    frame_dict[key] += setting_dict[key]
                else:
                    frame_dict[key] += setting_dict[key]
            for key in frame_dict.keys():
                if key not in setting_dict.keys():
                    frame_dict[key] += ['-']
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_colwidth', None)
    
    return pd.DataFrame(frame_dict)
        
        
def dataframe_edit(df, priorities):
    columns = list(df.columns)
    others = []
    for hyper in columns:
        if hyper not in priorities:
            others.append(hyper)
    columns = priorities + others

    df = df.reindex(columns=columns)
    df.insert(len(priorities), "-", ['||||||' for i in range(df.shape[0])], True)
    return df


def get_df_sample(df, sample_cond):
    cond = None
    for key in list(sample_cond.keys()):
        if key in df.columns.tolist():
            df = df[df[key] == sample_cond[key]]
        else:
            raise Exception('There is no {} key'.format(key))
            
    return df
        

def get_loss_filter(output, label, use_mask, args):
    pred = torch.argmax(output, dim=1)
    assert args.loss_filter in ['true_pos', 'pos', 'all', 'neg', None]
    if args.loss_filter == 'true_pos':
        loss_filter = label.type(torch.cuda.FloatTensor) * pred.type(torch.cuda.FloatTensor) * use_mask.type(torch.cuda.FloatTensor)
    elif args.loss_filter == 'pos':
        loss_filter = label.type(torch.cuda.FloatTensor) * use_mask.type(torch.cuda.FloatTensor)
    elif args.loss_filter == 'neg':
        loss_filter = (1-label.type(torch.cuda.FloatTensor)) * use_mask.type(torch.cuda.FloatTensor)
    elif args.loss_filter == 'all':
        loss_filter = use_mask.type(torch.cuda.FloatTensor)
    return loss_filter
        
        
        
        
        
        
        
        
        