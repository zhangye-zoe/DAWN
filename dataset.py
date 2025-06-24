"""
We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load (image, label_voronoi, label_cluster) items from given directory lists.

"""

import imp
import torch.utils.data as data
import os
from PIL import Image
import joblib
import numpy as np
from targets import gen_targets

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def img_loader(path, num_channels):
    # print('+' * 100)
    # print(path)
    if num_channels == 1:
        img = Image.open(path)
        if path.endswith("init_mask.png"):
            img = np.array(img)
            # print(np.unique(img))
            img[img>0] = 255
            # print(img.sum())
            img = Image.fromarray(img)

        if path.endswith("true_mask.png"):
            f_name = os.path.basename(path)
            # print(path)
            img = np.array(img)
            targets = gen_targets(img, (256,256))
            hv_map = targets["hv_map"] 
            # print(hv_map)
            """
            Version 1
            """
            hv_map = (hv_map+1)*127

            """
            Find change
            """
            # hv_map = hv_map + 1
            # np.save(f"hv_map/{f_name}.npy", hv_map)


            np_map = np.expand_dims(targets["np_map"],-1)
            np_map = np_map *255

            # print("hv_map", hv_map.shape)
            # print("np_map", np_map.shape)
            img = np.concatenate([hv_map, np_map], axis=-1).astype(np.uint8)
            # print(img.dtype)
            img = Image.fromarray(img)



            # print(path)
            # print(np.unique(np.array(img)))
        
    else:
        img = Image.open(path).convert('RGB')

    # print(path, np.array(img).shape)
    return img


# get the image list pairs
def get_imgs_list(dir_list, post_fix=None):
    """
    :param dir_list: [img1_dir, img2_dir, ...]
    :param post_fix: e.g. ['label_vor.png', 'label_cluster.png',...]
    :return: e.g. [(img1.png, img1_label_vor.png, img1_label_cluster.png), ...]
    """
    img_list = []
    if len(dir_list) == 0:
        return img_list
    if len(dir_list) != len(post_fix) + 1:
        raise (RuntimeError('Should specify the postfix of each img type except the first input.'))

    img_filename_list = [os.listdir(dir_list[i]) for i in range(len(dir_list))]

    for img in img_filename_list[0]:
        if not is_image_file(img):
            continue
        img1_name = os.path.splitext(img)[0]
        item = [os.path.join(dir_list[0], img),]
        for i in range(1, len(img_filename_list)):
            img_name = '{:s}_{:s}'.format(img1_name, post_fix[i-1])
            if img_name in img_filename_list[i]:
                img_path = os.path.join(dir_list[i], img_name)
                item.append(img_path)

        if len(item) == len(dir_list):
            img_list.append(tuple(item))

    return img_list


# dataset that supports multiple images
class DataFolder(data.Dataset):
    def __init__(self, dir_list, post_fix, num_channels, data_transform=None,dataset="MO", loader=img_loader):
        """
        :param dir_list: [img_dir, label_voronoi_dir, label_cluster_dir]
        :param post_fix:  ['label_vor.png', 'label_cluster.png']
        :param num_channels:  [3, 3, 3]
        :param data_transform: data transformations
        :param loader: image loader
        """
        super(DataFolder, self).__init__()
        if len(dir_list) != len(post_fix) + 1:
            raise (RuntimeError('Length of dir_list is different from length of post_fix + 1.'))
        if len(dir_list) != len(num_channels):
            raise (RuntimeError('Length of dir_list is different from length of num_channels.'))

        self.img_list = get_imgs_list(dir_list, post_fix)
        if len(self.img_list) == 0:
            raise(RuntimeError('Found 0 image pairs in given directories.'))

        self.data_transform = data_transform
        self.num_channels = num_channels
        self.loader = loader
        # self.det_count_file = joblib.load(f"../data/{dataset}/det_num_point.json")
        # self.seg_count_file = joblib.load(f"../data/{dataset}/seg_num_point.json")

    def __getitem__(self, index):
        
        img_paths = self.img_list[index]
        # print(img_paths)
        file_name = os.path.basename(img_paths[0])
        file = os.path.splitext(file_name)[0]
        # det_num = self.det_count_file[file[:-2]]
        # seg_num = self.seg_count_file[file[:-2]]
        sample = [self.loader(img_paths[i], self.num_channels[i]) for i in range(len(img_paths))]


        if self.data_transform is not None:
            # print(sample)
            sample = self.data_transform(sample)
            # print(sample[-1].sum())
        

        return sample

    def __len__(self):
        return len(self.img_list)

