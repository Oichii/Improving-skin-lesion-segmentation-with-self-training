from torch.utils.data import Dataset
import os
import cv2.cv2
import albumentations as albu
import torch
import numpy as np


class SkinSegmentationDataset(Dataset):
    def __init__(self, dataframe, root_dir='', img_path='', mask_path='', transforms=None, color_space='bgr'):
        self.df = dataframe

        self.img_path = os.path.join(root_dir, img_path)
        self.mask_path = os.path.join(root_dir, mask_path)

        self.colorSpace = color_space

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = albu.Compose([
                albu.Normalize()
            ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        image = cv2.cv2.imread(os.path.join(self.img_path, self.df.iloc[item, 0]))
        mask = cv2.cv2.imread(os.path.join(self.mask_path, self.df.iloc[item, 1]), 0)

        image = cv2.cv2.cvtColor(image, cv2.cv2.COLOR_BGR2RGB)

        transformed = self.transforms(image=image, mask=mask)

        image = transformed['image'].astype(np.float32)
        mask = transformed['mask'].astype(np.float32)

        image = image.transpose(2, 0, 1)
        image = image/255
        image = torch.tensor(image)

        mask = torch.tensor(mask)

        mask = mask/255
        mask = mask.unsqueeze(dim=0)

        return image, mask,  self.df.iloc[item, 0]


def get_transforms_segmentation(image_size, transform):
    if transform == "optical":
        t = albu.OpticalDistortion(distort_limit=1.0, p=0.7)
    elif transform == 'grid':
        t = albu.GridDistortion(num_steps=5, distort_limit=1., p=0.7)
    elif transform == 'elastic':
        t = albu.ElasticTransform(alpha=3, p=0.7)
    else:
        t = albu.CoarseDropout(max_height=int(image_size * 0.2), max_width=int(image_size * 0.2), max_holes=6, p=0.7)

    transforms_train = albu.Compose([
        albu.Transpose(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
        albu.OneOf([
            albu.MotionBlur(blur_limit=5),
            albu.MedianBlur(blur_limit=5),
            albu.GaussianBlur(blur_limit=(3, 5), sigma_limit=1),
            albu.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),
        t,
        albu.CLAHE(clip_limit=4.0, p=0.7),
        albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albu.Resize(image_size, image_size),
    ],
        additional_targets={"mask": "mask"}
    )

    transforms_val = albu.Compose([
        albu.Resize(image_size, image_size),
    ],
        additional_targets={"mask": "mask"}
    )

    return transforms_train, transforms_val

