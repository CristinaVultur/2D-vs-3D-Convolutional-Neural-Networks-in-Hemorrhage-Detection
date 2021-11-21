import pandas as pd
import os
import shutil

src_dir = '../rsna-intracranial-hemorrhage-detection/stage_2_train/'
dest_dir = '../rsna-intracranial-hemorrhage-detection/3d_train/'

scans = pd.read_csv('scans_reconstruction.csv')
print(scans.loc[scans.list_items.str.match('ID_000012eaf')]['list_id'])



labels = pd.read_csv("scans_with_lables.csv")
print(labels['any'].value_counts())
columSeries = labels[['scan_id']]

print()

"""for i in range(len(scans)):
    try:
        os.mkdir(dest_dir + scans.loc[i,'list_id'])
        shutil.copy(src_dir + scans.loc[i, "list_items"] + '.dcm', dest_dir+ scans.loc[i,'list_id'])
    except FileExistsError:
        shutil.copy(src_dir + scans.loc[i, "list_items"] + '.dcm', dest_dir+ scans.loc[i,'list_id'])"""


#shutil.copy(src_dir + value[0] +'.dcm', dest_dir)

for filename in os.listdir(src_dir):
    getName, exe = filename.split('.')
    n = not scans[scans.list_items.str.match(getName)].empty
    ok = [scans.loc[scans.list_items.str.match(getName), 'list_id']] in columSeries.values
    if n and ok:
        shutil.copy(src_dir + filename, dest_dir)


n = len(labels)
print(n)

merged_valid = labels[int(n*80/100):]
merged_train = labels[:int(n*80/100)]
print(merged_valid['any'].value_counts())
print(merged_train['any'].value_counts())

merged_valid.to_csv('valid3d.csv', index=False)
print(merged_valid['any'].value_counts())
merged_train.to_csv('train3d.csv', index=False)
print(merged_train['any'].value_counts())