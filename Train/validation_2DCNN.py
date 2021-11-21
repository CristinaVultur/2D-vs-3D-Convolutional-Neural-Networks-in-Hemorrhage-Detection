import pydicom
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import torch.optim as optim
from Datasets.TwoDDATA import TwoDCCNDataset
from sklearn.metrics import hamming_loss, accuracy_score, precision_score, recall_score, f1_score
from albumentations.pytorch import ToTensor

dicom = pydicom.dcmread('../rsna-intracranial-hemorrhage-detection/stage_2_train/' + 'ID_000012eaf' + '.dcm')

import albumentations
from torch.utils.data import DataLoader

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


train_images_dir = '../rsna-intracranial-hemorrhage-detection/stage_2_train/'
valid_dataset = TwoDCCNDataset(csv_file='../build/valid.csv',
                               path=train_images_dir,
                               labels=True,
                               transform=albumentations.Compose(
                                   [albumentations.Resize(256, 256),
                                    albumentations.Normalize(mean=[0.1738, 0.1433, 0.1970],
                                                             std=[0.3161, 0.2850, 0.3111],
                                                             max_pixel_value=1.),
                                    albumentations.HorizontalFlip(),
                                    albumentations.VerticalFlip(),
                                    albumentations.ShiftScaleRotate(),
                                    albumentations.RandomBrightnessContrast(),
                                    ToTensor()]))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data_loader_train = torch.utils.data.DataLoader(transformed_dataset, batch_size=32, shuffle=False)
data_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)


def my_oposite(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j] == 0:
                x[i][j] = 1
            else:
                x[i][j] = 0

    return x


batch_loss = 0
total_t = 0
correct_t = 0
aucs = []
losses = []
negative_recalls = []
accuracies = []
precisions = []
recalls = []
f1s = []
hammings = []
hamming_scores = []
predictions = []
labels = []
total_accuracies = []
total_losses = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
valid_loss_min = np.Inf
total_hamming_scores = []
total_hammings = []
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

with torch.no_grad():
    model.eval()
    for batch_id, batch_t in enumerate(data_loader_valid):
        data_t = batch_t["image"]
        target_t = batch_t["labels"]

        data_t = data_t.to(device, dtype=torch.float)
        target_t = target_t.to(device, dtype=torch.float)

        data_t, target_t = data_t.to(device), target_t.to(device)
        outputs_t = model(data_t)

        loss_t = criterion(outputs_t, target_t)
        losses.append(loss_t.item())
        target_t = target_t.cpu()
        outputs_t = outputs_t.cpu()
        labels.extend(target_t.tolist())

        hamming_lossv = hamming_loss(target_t, outputs_t.round())
        hammings.append(hamming_lossv)
        accuracy = accuracy_score(target_t, outputs_t.round())
        accuracies.append(accuracy)
        hamming_scorev = hamming_score(target_t, outputs_t.round())

        hamming_scores.append(hamming_scorev)
        total_accuracies.append(accuracy)
        total_losses.append(loss_t.item())
        total_hamming_scores.append(hamming_scorev)
        total_hammings.append(hamming_lossv)

        precisions.append(precision_score(target_t, outputs_t.round(), average='macro', zero_division=1))
        f1s.append(f1_score(target_t, outputs_t.round(), average='macro', zero_division=1))
        recalls.append(recall_score(target_t, outputs_t.round(), average='macro', zero_division=1))

        negative_recalls.append(
            recall_score(my_oposite(target_t), my_oposite(outputs_t.round()), average='macro', zero_division=1))

        if batch_id % 10 == 0:
            print(f'\nValidation batch: {batch_id}')
            print(f'Validation loss: {loss_t}')
            print(hamming_scorev, hamming_lossv, accuracy, f1s[batch_id], precisions[batch_id], recalls[batch_id],
                  negative_recalls[batch_id])

    # auc = np.array(aucs).mean()
    accuracy = np.array(accuracies).mean()
    batch_loss = np.array(losses).mean()
    h_loss = np.array(hammings).mean()
    h_score = np.array(hamming_scores).mean()
    precision = np.array(precisions).mean()
    f1 = np.array(f1s).mean()
    recall = np.array(recalls).mean()
    negative_recall = np.array(negative_recalls).mean()
    network_learned = batch_loss < valid_loss_min
    print(f'validation loss: {batch_loss:.4f}, validation acc: {(accuracy):.4f}\n')
    print(h_loss, h_score)
    print(precision, f1, recall, negative_recall)

