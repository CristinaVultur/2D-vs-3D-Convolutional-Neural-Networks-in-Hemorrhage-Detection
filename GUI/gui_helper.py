import pydicom
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from sklearn.metrics import hamming_loss, accuracy_score, precision_score, recall_score, f1_score
from albumentations.pytorch import ToTensor
import pandas as pd
from Datasets.TwoDDATA import bsb_window
import albumentations

def my_oposite(x):
    c = x
    for i in range(len(c)):
        for j in range(len(c[i])):
            if c[i][j] == 0:
                c[i][j] = 1
            else:
                c[i][j] = 0

    return c

def hamming_score(y_true, y_pred):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / \
                    float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)

    return np.mean(acc_list)


path = '../resnet_1.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 6),
    nn.Sigmoid()
)
load = torch.load(path)
model.load_state_dict(load)
model = model.cuda() if device else model
model.to(device)

path = '../rsna-intracranial-hemorrhage-detection/stage_2_train/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

valid = 'valid.csv'


def make_prediction(dcm, csv_file):
    data = pd.read_csv(csv_file)
    index = data.index
    condition = data["Image"] == dcm
    id = index[condition]
    id = id.to_list()
    idx = id[0]
    transform = albumentations.Compose(
        [albumentations.Resize(256, 256),
         albumentations.Normalize(mean=[0.1738, 0.1433, 0.1970], std=[0.3161, 0.2850, 0.3111],
                                  max_pixel_value=1.),
         albumentations.HorizontalFlip(),
         albumentations.VerticalFlip(),
         albumentations.ShiftScaleRotate(),
         albumentations.RandomBrightnessContrast(),
         ToTensor()])

    try:
        dicom = pydicom.dcmread(path + dcm + '.dcm')
        img = bsb_window(dicom)
    except:
        img = np.zeros((512, 512, 3))

    augmented = transform(image=img)
    img = augmented['image']

    vol = torch.unsqueeze(img, 0)

    labels = torch.tensor(data.loc[
                              idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural',
                                    'any']])
    labels = torch.unsqueeze(labels, 0)
    criterion = torch.nn.BCELoss()

    with torch.no_grad():
        model.eval()

        data_t = vol
        target_t = labels

        data_t = data_t.to(device, dtype=torch.float)
        target_t = target_t.to(device, dtype=torch.float)

        data_t, target_t = data_t.to(device), target_t.to(device)
        outputs_t = model(data_t)

        loss_t = criterion(outputs_t, target_t)

        target_t = target_t.cpu()
        outputs_t = outputs_t.cpu()

        h_loss = hamming_loss(target_t, outputs_t.round())

        accuracy = accuracy_score(target_t, outputs_t.round())

        h_score = hamming_score(target_t, outputs_t.round())

        precision = (precision_score(target_t, outputs_t.round(), average='macro', zero_division=1))
        f1 = (f1_score(target_t, outputs_t.round(), average='macro', zero_division=1))
        recall = recall_score(target_t, outputs_t.round(), average='macro', zero_division=1)

        copy_t = np.array(target_t).copy()

        copy_o = np.array(outputs_t.round()).copy()
        negative_recall = recall_score(my_oposite(target_t), my_oposite(outputs_t.round()), average='macro',
                                       zero_division=1)

        return copy_t, copy_o, loss_t.item(), accuracy, h_score, h_loss, precision, f1, recall, negative_recall