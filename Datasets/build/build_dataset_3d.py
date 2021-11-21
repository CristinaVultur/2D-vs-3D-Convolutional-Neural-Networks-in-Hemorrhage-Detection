import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
import csv

from TreeDData_Preprocessing import get_pixels_hu, load_scan, normalize, zero_center
# import gdcm

from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.morphology import disk, opening, closing
from tqdm import tqdm

from IPython.display import HTML
from PIL import Image
from TreeDData_Preprocessing import get_pixels_hu
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


dir_csv = '../rsna-intracranial-hemorrhage-detection'
train = pd.read_csv(os.path.join(dir_csv, 'stage_2_train.csv'))

train = pd.read_csv('scans_with_lables.csv')


train_images_dir = '../rsna-intracranial-hemorrhage-detection/stage_2_train/'
reconstructions = 'scans_reconstruction.csv'


# Read metadata for train/validation split
#reconstructions_pd = pd.read_csv(reconstructions)
print(train[train.scan_id.str.match('ID_0ab5820b2a')].values[0][0])
#
# Loop over the image files and store everything into a list.
#

"""merged_train = pd.merge(left=train, right=reconstructions_pd, how='left', left_on='scan_id', right_on='list_id')

print(merged_train)"""

with open('scans_reconstruction.csv') as csv_file:
    train_scans = {}
    train_scan_idx = []
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            continue
        else:
            slice = row[1]
            scan = row[0]
            try:
                train_scans[scan].append(slice)
            except:
                if not train[train.scan_id.str.match(scan)].empty:
                    train_scans[scan] = [slice]
                    train_scan_idx.append(scan)


max_slices = 0
for key in list(train_scans.keys()):
    if len(train_scans[key]) > max_slices:
        max_slices = len(train_scans[key])
        max_idx = key


print(max_slices)
n=len(train_scan_idx)

# adding padding slices to the scans so they have the same size for the cnn
print(max_idx)


pad = len(train_scans[max_idx])
for key in list(train_scans.keys()):
    for i in range(pad-len(train_scans[key])):
        train_scans[key].append('Pad')


"""#train_scan_pd = pd.DataFrame.from_dict(train_scans)
#train_scan_pd.to_csv('train_with_padding.csv', index=False)
with open('building_3d.csv', 'w', newline='') as f:  # Python 3
    w = csv.writer(f)
    first_row = ['scan_id', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
    for i in range(max_slices):
        first_row.append('image_'+str(i))
    w.writerow(first_row)
    for key, items in train_scans.items():
        row = []
        labels = train[train.scan_id.str.match(key)].values[0]
        row.append(labels[0]) #scan goes first
        for i in range(1, len(labels)):
            row.append(int(labels[i])) #labels as int
        for item in items:
            row.append(item)
        w.writerow(row)"""

"""with open('scans_with_lables.csv', 'w', newline='') as f:  # Python 3
    w = csv.writer(f)
    w.writerow(['scan_id', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any'])
    for key, items in train_scans.items():
        labels = np.zeros(6)
        for item in items:
            try:
                labels += np.array(train.loc[train.Image.str.match(item),
                                                    ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid',
                                                     'subdural', 'any']])[0]
            except:
                print(item)
        correct_labels = np.zeros(6)
        for i in range(len(labels)):
            if labels[i] > 0:
                correct_labels[i] = 1
        w.writerow([key, correct_labels[0], correct_labels[1], correct_labels[2], correct_labels[3], correct_labels[4],
                    correct_labels[5]])
"""

"""train_scan_pd = pd.DataFrame.from_dict(train_scans)

merged_train = pd.merge(left=train, right=train_scan_pd, how='left', left_on='scan_id', right_on='list_items')
n = len(merged_train)
print(merged_train)

merged_valid = train_scan_idx[int(n * 80 / 100):]
merged_train = train_scan_idx[:int(n * 80 / 100)]
merged_train.to_csv('train.csv', index=False)
print(merged_train['any'].value_counts())
merged_valid.to_csv('valid.csv', index=False)
print(merged_valid['any'].value_counts())"""
"""
COLS = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

# Prepare train table
train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)
train = train[['Image', 'Diagnosis', 'Label']]
train.drop_duplicates(inplace=True)
train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
train['Image'] = 'ID_' + train['Image']

# Remove invalid PNGs
png = glob.glob(os.path.join(train_images_dir, '*.dcm'))
png = [os.path.basename(png)[:-4] for png in png]
png = np.array(png)

train = train[train['Image'].isin(png)]

merged_train = pd.merge(left=train, right=train_metadata_noidx, how='left', left_on='Image', right_on='list_items')
n = len(merged_train)
print(n)

print(merged_train)

with open('scans_with_lables.csv', 'w', newline='') as f:  # Python 3
    w = csv.writer(f)
    w.writerow(['scan_id', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any'])
    for key, items in train_scans.items():
        labels = np.zeros(6)
        for item in items:
            try:
                labels += np.array(merged_train.loc[merged_train.Image.str.match(item),
                                                    ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid',
                                                     'subdural', 'any']])[0]
            except:
                print(item)
        correct_labels = np.zeros(6)
        for i in range(len(labels)):
            if labels[i] > 0:
                correct_labels[i] = 1
        w.writerow([key, correct_labels[0], correct_labels[1], correct_labels[2], correct_labels[3], correct_labels[4],
                    correct_labels[5]])


merged_valid = merged_train[int(n * 80 / 100):]
merged_train = merged_train[:int(n * 80 / 100)]


merged_train.to_csv('train.csv', index=False)
print(merged_train['any'].value_counts())
merged_valid.to_csv('valid.csv', index=False)
print(merged_valid['any'].value_counts())
"""
