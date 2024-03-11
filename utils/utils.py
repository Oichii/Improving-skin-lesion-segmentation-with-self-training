import os.path
import random

import cv2.cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def visualize_data(training_data):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        print(img, label)
        print("size: ", img.size())
        img = img.permute(1, 2, 0)
        img = img.squeeze().numpy()
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img, cmap="gray")
    wandb.log({"input images": wandb.Image(plt)})
    plt.show()


def visualize_segmentation_data(training_data):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()

        img, mask, _ = training_data[sample_idx]

        img = img.permute(1, 2, 0)
        mask = mask.permute(1, 2, 0)

        img = img.squeeze().numpy()
        mask = mask.squeeze().numpy()
        mask = (mask*255).astype('uint8')

        combined = cv2.cv2.bitwise_and(img, img, mask=mask)

        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        plt.imshow(combined, cmap="gray")

    plt.show()


def normalize(img):
    return (img-np.min(img))/(np.max(img)-np.min(img))
