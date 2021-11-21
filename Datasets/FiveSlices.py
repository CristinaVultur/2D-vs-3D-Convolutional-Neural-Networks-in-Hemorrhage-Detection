from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
import pydicom
from Datasets.TwoDDATA import correct_dcm


class FiveSlicesDataset(Dataset):

    def __init__(self, merged_csv, path, transform=None):
        self.path = path
        self.data = pd.read_csv(merged_csv)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            predict = []
            slice = self.data.loc[idx, 'Image']
            scan = self.data.loc[idx, 'SeriesInstanceUID']
            slices = self.data[self.data['SeriesInstanceUID'] == scan].sort_values(by=['ImagePositionSpan', 'ImageId'])
            slices = slices.reset_index(drop=True)
            idx_list = slices.index[slices['Image'] == slice].to_list()
            new_idx = idx_list[0]

            # merge the 2 slices from the right, our slice, and the 2 slices from the left
            for i in range(-2, 3):
                if new_idx - i >= 0 and new_idx + i < len(self.data):
                    s = slices.loc[new_idx - i, 'Image']
                    dcm = pydicom.dcmread(self.path + s + '.dcm')
                    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
                        correct_dcm(dcm)
                    else:
                        x = 5
                    predict.append(dcm.pixel_array)
                else:
                    imgs = np.zeros((512, 512))
                    predict.append(imgs)


        except:
            predict = np.zeros((5, 512, 512), dtype=np.int16)
        augmented = []
        if self.transform:
            for i in range(len(predict)):
                augment = self.transform(image=predict[i])
                augmented.append(augment['image'])
            image = np.stack([s for s in augmented])
            vol = torch.from_numpy(image).float()
        else:
            image = np.stack([s for s in predict])
            vol = torch.from_numpy(image).float()
        vol = torch.unsqueeze(vol, 0)
        labels = torch.tensor(self.data.loc[
                                  idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural',
                                        'any']])
        out_lables = labels.float()
        return {'image': vol, 'labels': out_lables}
