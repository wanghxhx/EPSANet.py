import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image


class ReadTiff(object):
    def tif2PIL(self, img_path):
        self.img = Image.open(img_path)
        # self.img_norm = (self.img - np.min(self.img)) / (np.max(self.img) - np.min(self.img))  # 归一化 image
        # self.img_pil = Image.fromarray(np.uint8(255 * self.img_norm))

        return self.img


def rand_crop(image, label, height, width):
    crop_size = [height, width]
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = torchvision.transforms.functional.crop(image, i, j, h, w)
    label = torchvision.transforms.functional.crop(label, i, j, h, w)
    return image, label


class MyDataset(Dataset):
    def __init__(self, transform, input_path, label_path):
        self.input_path = input_path
        self.input_image = os.listdir(input_path)
        self.label_path = label_path
        self.label_image = os.listdir(label_path)
        self.transform = transform

    def __len__(self):
        return len(self.input_image)

    def __getitem__(self, index):
        # 根据图片的路径，获取 tif 图像并转换成 Tensor
        input_image_path = os.path.join(self.input_path, self.input_image[index]).replace("\\", "/")  # 合成文件夹图像的路径
        input_image = ReadTiff().tif2PIL(input_image_path)  # PIL 能处理的图像格式

        input_label_path = os.path.join(self.label_path, self.label_image[index]).replace("\\", "/")  # 合成文件夹图像的路径
        label_image = ReadTiff().tif2PIL(input_label_path)  # PIL 能处理的图像格式

        train_image, train_label = rand_crop(input_image, label_image, 256, 256)
        return self.transform(train_image), self.transform(train_label)


class TestDataset(Dataset):
    def __init__(self, transform, input_path, label_path):
        self.input_path = input_path
        self.input_image = os.listdir(input_path)
        self.label_path = label_path
        self.label_image = os.listdir(label_path)
        self.transform = transform

    def __len__(self):
        return len(self.input_image)

    def __getitem__(self, index):
        # 根据图片的路径，获取 tif 图像并转换成 Tensor
        input_image_path = os.path.join(self.input_path, self.input_image[index]).replace("\\", "/")  # 合成文件夹图像的路径
        input_image = ReadTiff().tif2PIL(input_image_path)  # PIL 能处理的图像格式

        input_label_path = os.path.join(self.label_path, self.label_image[index]).replace("\\", "/")  # 合成文件夹图像的路径
        label_image = ReadTiff().tif2PIL(input_label_path)  # PIL 能处理的图像格式

        return self.transform(input_image), self.transform(label_image)
