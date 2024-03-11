"""
Script to prepare training csv from ISIC data.
"""
import pandas as pd
import os

data_2018 = pd.read_csv('../data/labels/ISIC2018_Task3_Training_GroundTruth.csv')
data_2018['diagnosis'] = 'unknown'
data_2018.loc[data_2018['MEL'] == 1.0, 'diagnosis'] = 4
data_2018.loc[data_2018['NV'] == 1.0, 'diagnosis'] = 5
data_2018.loc[data_2018['BCC'] == 1.0, 'diagnosis'] = 1
data_2018.loc[data_2018['AKIEC'] == 1.0, 'diagnosis'] = 0
data_2018.loc[data_2018['BKL'] == 1.0, 'diagnosis'] = 2
data_2018.loc[data_2018['DF'] == 1.0, 'diagnosis'] = 3
data_2018.loc[data_2018['VASC'] == 1.0, 'diagnosis'] = 7

data_2017 = pd.read_csv('../data/labels/ISIC-2017_Training_Part3_GroundTruth.csv')
data_2017_val = pd.read_csv('../data/labels/ISIC-2017_Validation_Part3_GroundTruth.csv')
data_2017_test = pd.read_csv('../data/labels/ISIC-2017_Test_v2_Part3_GroundTruth.csv')

data_2018_val = pd.read_csv('../data/labels/ISIC2018_Task3_Validation_GroundTruth.csv')
all_data = pd.read_csv('../data/labels/ISIC_2019_2020.csv')

available_masks = os.listdir('../data/mask')

segmentation_data = []
i = 0
for mask in available_masks:
    name, ext = os.path.splitext(mask)
    name = name[:-13]
    data = all_data[all_data['image_path'] == name]
    # print(data['binary_target'],  data['target'])
    data1 = data_2018[data_2018['image'] == name]
    data2 = data_2017[data_2017['image_id'] == name]
    data3 = data_2017_val[data_2017_val['image_id'] == name]
    data4 = data_2018_val[data_2018_val['image'] == name]
    data5 = data_2017_test[data_2017_test['image_id'] == name]

    if not data.empty:
        if os.path.exists(os.path.join('../data/images', name + '.jpg')):
            segmentation_data.append({'image_path': name, 'binary_target': data['binary_target'].item(),
                                      'target': data['target'].item()})
    elif not data1.empty:
        if os.path.exists(os.path.join('../data/images', name + '.jpg')):
            segmentation_data.append({'image_path': name, 'binary_target': data1['MEL'].item(),
                                      'target': data1['diagnosis'].item()})
    elif not data2.empty:
        target = 1 if data2['melanoma'].item() > 0 else 2
        if os.path.exists(os.path.join('../data/images', name + '.jpg')):
            segmentation_data.append({'image_path': name, 'binary_target': data2['melanoma'].item(),
                                      'target': target})
    elif not data3.empty:
        target = 1 if data3['melanoma'].item() > 0 else 2
        if os.path.exists(os.path.join('../data/images', name + '.jpg')):
            segmentation_data.append({'image_path': name, 'binary_target': data3['melanoma'].item(),
                                      'target': target})
    elif not data4.empty:
        if os.path.exists(os.path.join('../data/images', name + '.jpg')):
            segmentation_data.append({'image_path': name, 'binary_target': data4['MEL'].item(),
                                      'target': data4['diagnosis'].item()})
    elif not data5.empty:
        target = 1 if data5['melanoma'].item() > 0 else 2
        if os.path.exists(os.path.join('../data/images', name + '.jpg')):
            segmentation_data.append({'image_path': name, 'binary_target': data5['melanoma'].item(),
                                      'target': target})
    else:
        i += 1
        print(name, i)

segmentation_data = pd.DataFrame(segmentation_data)

tfrecord2fold = {
    8: 0, 5: 0, 11: 0,
    7: 1, 0: 1, 6: 1,
    10: 2, 12: 2, 13: 2,
    9: 3, 1: 3, 3: 3,
    14: 4, 2: 4, 4: 4,
}
segmentation_data['tfrecord'] = segmentation_data.index % 15
segmentation_data['fold'] = segmentation_data['tfrecord'].map(tfrecord2fold)

segmentation_data.to_csv('../data/labels/segmentation.csv', index=False)
