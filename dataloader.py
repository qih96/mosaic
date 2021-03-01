from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFilter
import numpy as np
import random, torch
from copy import deepcopy
import math

def image_train(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize(resize_size),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
    return transforms.Compose([
    transforms.Resize(resize_size),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    normalize
    ])

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def mosaic_aug(img, num_patch_1, num_patch_2, fix_p):
    if num_patch_1 != 1: 
        temp = deepcopy(img)
        n = num_patch_1
        index1 = np.random.permutation(n)
        index2 = np.random.permutation(n)
        for i in range(n):
            for j in range(n):
                img[i::n, j::n, :] = temp[index1[i]::n, index2[j]::n, :]

    if num_patch_2 != 1:
        temp = deepcopy(img)
        n = num_patch_2
        patch_size = 224 // n
        patch_size *= fix_p
        index1 = np.random.permutation(n//fix_p)
        index2 = np.random.permutation(n//fix_p)

        # extra_i, extra_j = random.choices([0, 1], k=2)
        if n % fix_p != 0:
            extra_i, extra_j = random.choices([i for i in range(math.ceil(n/fix_p))], k=2)
            extra_size = patch_size // fix_p
        else:
            extra_i = extra_j = 0
            extra_size = 0
        
        for i in range(n//fix_p):
            for j in range(n//fix_p):
                index_w_1 = patch_size*i if i < extra_i else patch_size*i+extra_size
                index_h_1 = patch_size*j if j < extra_j else patch_size*j+extra_size

                index_w_2 = patch_size*index1[i] if index1[i] < extra_i else patch_size*index1[i]+extra_size
                index_h_2 = patch_size*index2[j] if index2[j] < extra_j else patch_size*index2[j]+extra_size
                patch = temp[index_w_2:index_w_2+patch_size, index_h_2:index_h_2+patch_size, :]
                # if random.random() < 0.5:
                #     # patch = torch.flip(patch, dims=[2])
                #     patch = patch[:, ::-1, :]
                img[index_w_1:index_w_1+patch_size, index_h_1:index_h_1+patch_size, :] = patch

def mosaic_aug_v2(img, num_patch, p=0.5):
    # if num_patch_1 != 1: 
    #     temp = deepcopy(img)
    #     n = num_patch_1
    #     index1 = np.random.permutation(n)
    #     index2 = np.random.permutation(n)
    #     for i in range(n):
    #         for j in range(n):
    #             img[i::n, j::n, :] = temp[index1[i]::n, index2[j]::n, :]

    # if num_patch_2 != 1:
    temp = deepcopy(img)

    patch_size = 224 // num_patch
    # index1 = np.random.permutation(n//fix_p)
    # index2 = np.random.permutation(n//fix_p)

    # extra_i, extra_j = random.choices([0, 1], k=2)
    extra_size = 224 % num_patch
    if 224 % num_patch != 0:
        extra_i, extra_j = random.choices([i for i in range(num_patch)], k=2)
    else:
        extra_i = extra_j = 0

    for i in range(num_patch):
        for j in range(num_patch):
            if random.random() < p:
                continue
            index_w_1 = patch_size*i if i < extra_i else patch_size*i+extra_size
            index_h_1 = patch_size*j if j < extra_j else patch_size*j+extra_size

            # index_w_2 = patch_size*index1[i] if index1[i] < extra_i else patch_size*index1[i]+extra_size
            # index_h_2 = patch_size*index2[j] if index2[j] < extra_j else patch_size*index2[j]+extra_size
            index1 = np.random.randint(0, num_patch)
            index2 = np.random.randint(0, num_patch)
            index_w_2 = patch_size*index1 if index1 < extra_i else patch_size*index1+extra_size
            index_h_2 = patch_size*index2 if index2 < extra_j else patch_size*index2+extra_size
            patch = temp[index_w_2:index_w_2+patch_size, index_h_2:index_h_2+patch_size, :]

            img[index_w_1:index_w_1+patch_size, index_h_1:index_h_1+patch_size, :] = patch

def mosaic_aug_v3(img, num_patch):

    temp = deepcopy(img)
    random_i = np.random.randint(0,1)
    random_j = np.random.randint(0,1)
    # print(random_i, random_j)

    patch_size = 224 // num_patch
    half_patch_size = patch_size // 2
    for i in range(1, num_patch-1):
        for j in range(1, num_patch-1):
            # print(i*patch_size, j*patch_size, random_i, random_j)
            # print(temp[(i-1)*patch_size:(i+2)*patch_size, (j-1)*patch_size:(j+2)*patch_size, :][random_i::3,random_j::3,:].shape, img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :].shape)
            img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :] = \
            temp[(i)*patch_size-half_patch_size:(i+1)*patch_size+half_patch_size, (j)*patch_size-half_patch_size:(j+1)*patch_size+half_patch_size, :][random_i::2,random_j::2,:]

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'
                 ,pseudo=False, params=None):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("No image found !"))

        self.imgs = imgs
        self.transform = transform
        self.transforms_1 = transforms.Compose(transform.transforms[:-2])
        self.transforms_2 = transforms.Compose(transform.transforms[-2:])
        self.target_transform = target_transform
        self.pseudo=pseudo
        try:
            self.params = params
            self.num_patch_1 = params.mosaic_1
            self.num_patch_2 = params.mosaic_2
        except:
            self.num_patch_1 = self.num_patch_2 = None
        self.aux_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomAffine(45, translate=None, scale=None, shear=10, resample=False, fillcolor=0),
                transforms.RandomCrop(224),
                # transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
        
        self.augmentation_w = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip(),
            ])

        self.augmentation_s = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip(),
            ])

        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        if not self.pseudo:
            path, target = self.imgs[index]
            or_img = self.loader(path)
            if self.transform is not None:
                # img = self.transform(or_img)
                temp_img = self.transforms_1(or_img)
                img = self.transforms_2(temp_img)
                
                temp_img_ = self.transforms_1(or_img)
                img_ = self.transforms_2(temp_img_)
                # img_ = image_test()(or_img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            if self.num_patch_1 != None or self.num_patch_2 != None:
                # temp_img = self.augmentation(temp_img)
                # mosaic_w = np.array(temp_img)
                mosaic_w = np.array(self.augmentation_w(or_img))
                temp_img = np.concatenate((temp_img, mosaic_w), axis=0)
                # [112 74 56 32]
                # mosaic_aug_v2(mosaic_w, 1, self.num_patch_2, 74, p=0.0)
                mosaic_aug_v2(mosaic_w, self.num_patch_1, p=0.)

                temp_img_ = np.concatenate((mosaic_w, mosaic_w), axis=0)
                mosaic_w = self.transforms_2(mosaic_w)

                # temp_img = self.augmentation(temp_img)
                # mosaic_s = np.array(temp_img)
                mosaic_s = np.array(self.augmentation_s(or_img))
                temp_img = np.concatenate((temp_img, mosaic_s), axis=0)

                # mosaic_aug_v2(mosaic_s, 1, self.num_patch_2, 56, p=0.)
                mosaic_aug_v2(mosaic_s, self.num_patch_2, p=0.)

                temp_img_ = np.concatenate((temp_img_, mosaic_s), axis=0)
                Image.fromarray(np.concatenate((temp_img, temp_img_), axis=1).astype('uint8')).convert('RGB').save('{}/test.jpg'.format(self.params.save_dir))
                mosaic_s = self.transforms_2(mosaic_s)


                return img, img_, mosaic_w, mosaic_s, target
            else:
                return img, target
        else:
            path, label, weight = self.imgs[index]
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)

            return img,  label, weight

    def __len__(self):
        return len(self.imgs)
def make_dataset(image_list, labels):

    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0],int(val.split()[1]),float(val.split()[2])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')