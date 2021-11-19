from __future__ import print_function
import glob
import os
import pandas as pd
import numpy as np

dir_csv = '../rsna-intracranial-hemorrhage-detection'
test_images_dir = '../rsna-intracranial-hemorrhage-detection/stage_2_test/'
train_images_dir = '../rsna-intracranial-hemorrhage-detection/stage_2_train/'
train_metadata_csv = '../rsna-intracranial-hemorrhage-detection/train_metadata_noidx.csv'
test_metadata_csv = '../rsna-intracranial-hemorrhage-detection/test_metadata_noidx.csv'

train = pd.read_csv(os.path.join(dir_csv,'stage_2_train.csv'))
test = pd.read_csv(os.path.join(dir_csv, 'stage_2_sample_submission.csv'))

# Read metadata for train/validation split
test_metadata_noidx = pd.read_csv(test_metadata_csv)
train_metadata_noidx = pd.read_csv(train_metadata_csv)


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
merged_train = pd.merge(left=train, right=train_metadata_noidx, how='left', left_on='Image', right_on='ImageId')
train_series = train_metadata_noidx['SeriesInstanceUID'].unique()

n = len(train_series)

merged_train.to_csv('five_train.csv', index = False)
valid_series = train[int(n*80/100):]
train_series = train[:int(n*80/100)]


"""#train.to_csv('train.csv', index=False)
print(train['any'].value_counts())
print(train['epidural'].value_counts())
print(train['intraparenchymal'].value_counts())
print(train['intraventricular'].value_counts())
print(train['subarachnoid'].value_counts())
print(train['subdural'].value_counts())
#valid.to_csv('valid.csv', index=False)
#print(valid['any'].value_counts())
valid = pd.read_csv('valid.csv')
print(valid['any'].value_counts())
print(valid['epidural'].value_counts())
print(valid['intraparenchymal'].value_counts())
print(valid['intraventricular'].value_counts())
print(valid['subarachnoid'].value_counts())
print(valid['subdural'].value_counts())"""

