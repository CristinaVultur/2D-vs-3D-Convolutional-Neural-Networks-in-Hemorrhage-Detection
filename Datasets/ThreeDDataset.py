from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
from TreeDData_Preprocessing import load_scan, normalize, zero_center


class ThreeDCCNDataset(Dataset):

    def __init__(self, csv_file, path, transform=None):
        self.path = path
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            # reconstruct the whole scan
            slices = []
            for i in range(60):
                slices.append(self.data.loc[idx, 'image_' + str(i)])
            img = load_scan(scan=slices)
            img = normalize(img)
            img = zero_center(img)

        except:
            img = np.zeros((60, 512, 512), dtype=np.int16)
        if self.transform:
            for i in img:
                augmented = self.transform(image=i)
                i = augmented['image']

        vol = torch.from_numpy(img).float()
        vol = torch.unsqueeze(vol, 0)

        labels = torch.tensor(self.data.loc[
                                  idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural',
                                        'any']])
        out_lables = labels.float()
        return {'image': vol, 'labels': out_lables}
