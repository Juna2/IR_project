import numpy as np
import collections
import h5py, ntpath
import medpy, medpy.io
from PIL import Image
from PIL import Image, ImageDraw

import os, os.path
import skimage, skimage.transform
import albumentations as albu




def extract_samples(data, image_path, label_path):
    image_data, _ = medpy.io.load(image_path)
    image_data = image_data.transpose(2,0,1)
    seg_data, _ = medpy.io.load(label_path)
    seg_data = seg_data.transpose(2,0,1)
    labels = seg_data.sum((1,2)) > 1

    print (collections.Counter(labels))

    for i in range(image_data.shape[0]):
        data.append((image_data[i],seg_data[i],labels[i]))

def extract_samples2(data, labels, image_path, label_path):
    new_size = 224
    image_data, _ = medpy.io.load(image_path)
    print('image_data :', image_data.shape)
    image_data = image_data.transpose(2,0,1)
    seg_data, _ = medpy.io.load(label_path)
    seg_data = seg_data.transpose(2,0,1)
    these_labels = seg_data.sum((1,2)) > 1

    print(collections.Counter(these_labels))

    for i in range(image_data.shape[0]):
        
        # slice의 크기를 224x224로 맞춰준다.
        img_resize = skimage.transform.resize(image_data[i], (new_size, new_size))
        seg_resize = skimage.transform.resize(seg_data[i], (new_size, new_size))
        seg_resize = (seg_resize > 0) * 1.0
        
        data.append([img_resize, seg_resize])
        labels.append(these_labels[i])


def compute_hdf5(dataroot, files, hdf5_name):
    if not os.path.exists(os.path.split(hdf5_name)[0]):
        os.makedirs(os.path.split(hdf5_name)[0])
    
    # 학습할 때 입력한 경로에 dataset_in_use.hdf5를 만들기 시작
    with h5py.File(hdf5_name,"w") as hf:
        # files는 dataset.json 파일에 지정된 train을 위한 환자들의 파일들
        files = sorted(files, key=lambda k: k["image"])
        for i, p in enumerate(files):
            print(p["image"], p["label"])
            name = ntpath.basename(p["image"])

            grp = hf.create_group(name)
            grp.attrs['name'] = name
            grp.attrs['author'] = "jpc"

            samples = []
            labels = []

            extract_samples2(samples, labels, dataroot +'/'+ p["image"], dataroot +'/'+ p["label"])

            grp_slices = grp.create_group("slices")
            for idx, zlice in enumerate(samples):
                print(".", end=" ")
                grp_slices.create_dataset(str(idx),data=zlice, compression='gzip')
            print(".")
            grp.create_dataset("labels",data=labels)



def wrap_setattr(attr, value):
    def foo(func):
        setattr(func, attr, value)
        return func
    return foo

def setdatasetname(value):
    return wrap_setattr('_DG_NAME', value)




def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def to_tensor_mask(x, **kwargs):
    x[x>=0.5] = 1.0
    x[x<0.5] = 0.0
    return x.transpose(2, 0, 1).astype('float32')


def data_preprocessing():
    transform = [
        albu.Lambda(image=to_tensor, mask=to_tensor_mask),
    ]
    return albu.Compose(transform)

def data_augmentation():
    transform = [
        
        albu.Resize(244, 244),
        albu.RandomCrop(height=224, width=224, always_apply=True),
        albu.VerticalFlip(p=0.5),
        albu.HorizontalFlip(p=0.5),
        
    ]
    return albu.Compose(transform)



class Patch():
    def __init__(self, img_size, trans):
        self.img_size = img_size
        self.color_cand = [
            (255, 0, 0, trans),
            (0, 255, 0, trans),
            (0, 0, 255, trans),
        ]
        min_height = int(img_size * 30/224)
        max_height = int(img_size * 32/224)
        min_width = int(img_size * 30/224)
        max_width = int(img_size * 32/224)

        h_thr = int(img_size * 32/224)
        w_thr = h_thr
        assert h_thr >= min_height
        assert w_thr >= min_width

        img_4_sample = np.ones([img_size, img_size], dtype=np.uint8)
        img_4_sample[:min_height, :] = 0
        img_4_sample[max_height:, :] = 0
        img_4_sample[:, :min_width] = 0
        img_4_sample[:, max_width:] = 0
        img_4_sample[h_thr:,w_thr:] = 0
        
        # print('하얀 부분에서 ellipse를 정의하기 위한 height, width를 sample한다')
        # plt.imshow(img_4_sample, cmap='gray')
        # plt.show()

        self.height_cand, self.width_cand = np.where(img_4_sample == 1)
        
        # print('height, width :', height, width)
        # print(img_4_sample[height, width])
        
    def sample_center(self,):
        offset = int(np.random.rand(1)*224)
        center_cand = [
            [offset, 0],
            [0, offset],
            [self.img_size, offset],
            [offset, self.img_size],
        ]
        return center_cand[int(np.random.rand(1)*4)]

    def sample_ellipse(self,):
        idx = np.random.choice(np.arange(len(self.height_cand)), 1)[0]
        height, width = self.height_cand[idx], self.width_cand[idx]
        return height, width
    
    def sample_color(self,):
        return self.color_cand[np.random.choice(np.arange(len(self.color_cand), dtype=np.int), 1)[0]]

    def put_patch(self, img):
        height, width = self.sample_ellipse()
        center = self.sample_center()
        color = self.sample_color()
        # print('center :', center)

        ell_def = [center[0]-width, center[1]-height, center[0]+width, center[1]+height]

        Img = Image.fromarray(img)
        draw = ImageDraw.Draw(Img, 'RGBA')
        draw.ellipse(ell_def, fill=color, outline=None)

        img = np.array(Img)
        return img










