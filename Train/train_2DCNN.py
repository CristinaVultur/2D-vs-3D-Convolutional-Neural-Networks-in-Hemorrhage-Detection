from Datasets.TwoDDATA import TwoDCCN_Dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.optim as optim
from skimage.transform import histogram_matching
import albumentations
from torchvision import transforms
from albumentations.pytorch import ToTensor
import pylab
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def Accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        max_true = np.array(y_true[i][0:len(y_true[i])-1]).argmax()
        max_pred = np.array(y_pred[i][0:len(y_pred[i])-1]).argmax()
        if max_true == max_pred:
            temp+=1

    return temp / y_true.shape[0]

saved_model_dir = '../2DModel_checkpoint/'
train_images_dir = '../rsna-intracranial-hemorrhage-detection/stage_2_train/'
test_images_dir = '../rsna-intracranial-hemorrhage-detection/stage_2_test/'

transformed_dataset = TwoDCCN_Dataset(csv_file='train.csv',
                                       path=train_images_dir,
                                        labels=True,
                                        transform=albumentations.Compose(
                                               [albumentations.Resize(256, 256),
                                                albumentations.Normalize(mean=[0.1738, 0.1433, 0.1970], std=[0.3161, 0.2850, 0.3111],
                                                          max_pixel_value=1.),
                                                albumentations.HorizontalFlip(),
                                                albumentations.VerticalFlip(),
                                                albumentations.ShiftScaleRotate(),
                                                albumentations.RandomBrightnessContrast(),
                                               ToTensor()]))
valid_dataset= TwoDCCN_Dataset(csv_file='valid.csv',
                                       path=train_images_dir,
                                        labels=True,
                                        transform=albumentations.Compose(
                                               [albumentations.Resize(256, 256),
                                                albumentations.Normalize(mean=[0.1738, 0.1433, 0.1970], std=[0.3161, 0.2850, 0.3111],
                                                          max_pixel_value=1.),
                                                albumentations.HorizontalFlip(),
                                                albumentations.VerticalFlip(),
                                                albumentations.ShiftScaleRotate(),
                                                albumentations.RandomBrightnessContrast(),
                                               ToTensor()]))


print(len(transformed_dataset))
print(transformed_dataset[0]['image'].shape)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_loader_train = torch.utils.data.DataLoader(transformed_dataset, batch_size=32, shuffle=False)
data_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)


model = models.resnet18(pretrained=True)
model = model.cuda() if device else model
print(torch.cuda.is_available())
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)


use_cuda =  torch.cuda.is_available()

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 6),
    nn.Sigmoid()
)


model.fc = model.fc.cuda() if use_cuda else model.fc

model.to(device)

n_epochs = 10
print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(data_loader_train)

for epoch in range(1, n_epochs + 1):
    running_loss = 0.0
    correct = 0
    total = 0
    print(f'Epoch {epoch}\n')

    model.train()
    for batch_id, batch in enumerate(data_loader_train):
        batch_data = batch["image"]
        batch_labels = batch["labels"]

        batch_data = batch_data.to(device, dtype=torch.float)
        batch_labels = batch_labels.to(device, dtype=torch.float)

        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        optimizer.zero_grad()

        outputs_n = model(batch_data)
        print(outputs_n.type())
        print(batch_labels.type())
        loss = criterion(outputs_n, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_id) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch, n_epochs, batch_id, total_step, loss.item()))


    train_loss.append(running_loss / total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}')

    batch_loss = 0
    total_t = 0
    correct_t = 0
    losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    predictions = []
    labels = []
    with torch.no_grad():
        model.eval()
        for batch_id, batch_t in enumerate(data_loader_valid ):
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

            accuracy = Accuracy(target_t,outputs_t)
            accuracies.append(accuracy)

            if batch_id % 10 == 0:
                print(f'\nValidation batch: {batch_id}')
                print(f'Validation loss: {loss_t}')

        accuracy = np.array(accuracies).mean()
        batch_loss = np.array(losses).mean()
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {batch_loss:.4f}, validation acc: {(accuracy):.4f}\n')

        if network_learned:
            valid_loss_min = batch_loss
            torch.save(model.state_dict(), 'resnet_1.pt')
            print('Improvement-Detected, save-model')
    model.train()