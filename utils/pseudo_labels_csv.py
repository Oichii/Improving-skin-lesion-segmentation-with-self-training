"""
Script to prepare training csvs with pseudo-labels
"""
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import numpy as np

test_path = r'data\segmentation\test'
test_list = os.listdir(test_path)
test_list = [x.lower() for x in test_list]

train_path = r'data\segmentation\train'
train_list = os.listdir(train_path)
train_list = [x.lower() for x in train_list]

path = r'data\images'
all_images = os.listdir(path)

# sum of real ground truth masks and pseudo-masks
pseudo_labels_train = []

for img in all_images:
    img_mask = img.lower().replace('.jpg', '_segmentation.png')
    if img_mask in test_list:
        pass
    elif img_mask in train_list:
        pseudo_labels_train.append({'image': img,
                                    'mask': img.replace('.jpg', '_segmentation.png'), 'pseudo_label': False})
    else:
        mask = cv2.imread(os.path.join(r'pseudo_mask', img + '_mask.png'))
        mask = mask/255
        if np.sum(np.sum(mask)) < 100:
            print('mask too small')
            print(os.path.join(path, img), os.path.join('wrong_masks', img))
            shutil.copy(os.path.join(path, img), os.path.join('wrong_masks', img))
            image = cv2.imread(os.path.join(path, img))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (256, 256))
            plt.imshow(image)
            plt.imshow(mask, alpha=0.2)
            plt.savefig(os.path.join('small_masks_check', img))
            plt.clf()
        else:
            pseudo_labels_train.append({'image': img, 'mask': img.replace('.jpg', '_mask.png'), 'pseudo_label': True})

pseudo_labels_train_df = pd.DataFrame(pseudo_labels_train)
tfrecord2fold = {
    8: 0, 5: 0, 11: 0,
    7: 1, 0: 1, 6: 1,
    10: 2, 12: 2, 13: 2,
    9: 3, 1: 3, 3: 3,
    14: 4, 2: 4, 4: 4,
}
pseudo_labels_train_df['tfrecord'] = pseudo_labels_train_df.index % 15
pseudo_labels_train_df['fold'] = pseudo_labels_train_df['tfrecord'].map(tfrecord2fold)
pseudo_labels_train_df.to_csv(r'data\segmentation\pseudolabels_train.csv', index=False)
