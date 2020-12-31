import os
import cv2

import torch
from torch.utils.data import Dataset

from typing import *


class ImageDataSet(Dataset):

    IMAGE_FILES = [
        '.png'
        '.jpg',
        '.jpeg',
    ]

    @classmethod
    def image_files(cls) -> List[str]:
        return cls.IMAGE_FILES

    def __init__(self, train_dir: str):
        super().__init__()
        self.train_dir = train_dir

        image_names = os.listdir(self.train_dir)

        self.data = []
        for image_name in image_names:
            img = cv2.imread(f'{self.train_dir}/{image_name}')

            lr, hr = cv2.resize(img, (100, 100), cv2.INTER_CUBIC), cv2.resize(img, (400, 400))

            self.data.append((
                torch.from_numpy(lr).float(), torch.from_numpy(hr).float()
            ))

    def __getitem__(self, item: int) -> (torch.Tensor, torch.Tensor):
        return self.data[item]

    def __len__(self) -> int:
        return len(self.data)
