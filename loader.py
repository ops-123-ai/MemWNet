import os
import cv2
import numpy as np
import torch.utils.data as data
import torchvision
from models.common import config


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['.png', '.bmp', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


class TrainDatasetFromFolder(data.Dataset):
    def __init__(self, dataset_dir, block_size):
        super().__init__()
        self.image_filenames = []

        for path, _, file_list in os.walk(dataset_dir):
            for file_name in file_list:
                self.image_filenames.append(os.path.join(path, file_name))

        self.image_filenames *= 16

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.RandomCrop(block_size),
        ])

    def __getitem__(self, index):
        for i in range(index, len(self.image_filenames)):
            Img = cv2.imread(self.image_filenames[i], flags=1)
            Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
            one_image = np.float32(Img_yuv[:, :, 0]) / 255.

            if one_image.shape[0] >= config.para.patch_size and one_image.shape[1] >= config.para.patch_size:
                return self.transform(one_image)

    def __len__(self):
        return len(self.image_filenames)