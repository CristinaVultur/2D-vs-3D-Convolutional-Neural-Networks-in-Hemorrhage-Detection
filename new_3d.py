import torch
from MedicalNet.setting import parse_opts

from MedicalNet.model import generate_model
import os
from sklearn.metrics import classification_report
from FiveSlices import FiveSlices_Dataset
import torch.optim as optim

from sklearn.metrics import precision_score, confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss, accuracy_score, roc_auc_score
import albumentations

from torch.utils.data import DataLoader

import numpy as np
from albumentations.pytorch import ToTensor
import pylab
path = 'MedicalNet/MedicalNet_pytorch_files2/pretrain/resnet_18.pth'
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

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

"""

def train(data_loader, data_loader_valid, model, optimizer,ajust_lr, total_epochs, save_interval, save_folder, sets):
    # settings
    total_accuracies=[]
    total_losses=[]
    valid_loss_min = np.Inf
    total_hamming_scores=[]
    total_hammings=[]
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    criterion =torch.nn.BCELoss()

    print("Current setting is:")
    print(sets)
    print("\n\n")
    if not sets.no_cuda:
        criterion = criterion.cuda()

    model.train()
    train_time_sp = time.time()
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))

        rate = ajust_lr(optimizer, epoch)

        for batch_id, batch_data in enumerate(data_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes = batch_data['image']
            label = batch_data['labels']

            if not sets.no_cuda:
                volumes = volumes.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            out_masks = model(volumes)

            print(out_masks)
            print(label)

            loss_value_seg = criterion(out_masks, label)
            loss = loss_value_seg
            loss.backward()
            optimizer.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                'Batch: {}-{} ({}), loss = {:.3f}, avg_batch_time = {:.3f}' \
                    .format(epoch, batch_id, batch_id_sp/10, loss.item(), avg_batch_time))

            if not sets.ci_test:
                # save model
                if batch_id == 0 and batch_id_sp != 0:
                    # if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    model_save_path = '{}_epoch_{}_batch_{}.pth'.format(save_folder, epoch, batch_id)
                    print(model_save_path)
                    print(model_save_path)
                    model_save_dir = os.path.dirname(model_save_path)
                    if not os.path.exists(model_save_path):
                        os.makedirs(model_save_path)

                    log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id))
                    torch.save({
                        'ecpoch': epoch,
                        'batch_id': batch_id,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        model_save_path)

        batch_loss = 0
        total_t = 0
        correct_t = 0
        losses = []
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        hammings =[]
        hamming_scores = []
        predictions = []
        labels = []

        with torch.no_grad():
            model.eval()
            for batch_id, batch_t in enumerate(data_loader_valid):
                data_t = batch_t["image"]
                target_t = batch_t["labels"]

                if not sets.no_cuda:
                    data_t = data_t.cuda()
                    target_t = target_t.cuda()

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
                if batch_id % 10 == 0:
                    print(f'\nValidation batch: {batch_id}')
                    print(f'Validation loss: {loss_t}')
                    print(f'Hamming score: {hamming_scorev}')
                    print(f'Hamming loss: {hamming_lossv}')

            pickle.dump(total_accuracies, open('acc.pkl', 'wb'))
            pickle.dump(total_losses, open('losses.pkl', 'wb'))
            pickle.dump(total_hamming_scores, open('scores.pkl', 'wb'))
            pickle.dump(total_hammings, open('h_loss.pkl', 'wb'))
            accuracy = np.array(accuracies).mean()
            batch_loss = np.array(losses).mean()
            hamming_lossv = np.array(hammings).mean()
            hamming_scorev = np.array(hamming_scores).mean()

            print(f'validation loss: {batch_loss:.4f}, validation hamming loss: {(hamming_scorev):.4f}\n')

            network_learned = batch_loss < valid_loss_min
            print(f'validation loss: {batch_loss:.4f}, validation acc: {(accuracy):.4f}\n')

            if network_learned:
                valid_loss_min = batch_loss
                torch.save(model.state_dict(), 'resnet_3d_1.pt')
                print('Improvement-Detected, save-model')
        model.train()

    print('Finished training')
    print(losses,accuracies)
    return  total_losses,  total_accuracies,  total_hamming_scores,  total_hammings

"""
def my_oposite(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j]==0:
                x[i][j]=1
            else:
                x[i][j]=0

    return x
sets = parse_opts()
sets.gpu_id =  [torch.cuda.device(i) for i in range(torch.cuda.device_count())]


sets.n_epochs = 5
sets.no_cuda = False
sets.data_root = '../rsna-intracranial-hemorrhage-detection/stage_2_train/'
sets.pretrain_path = 'MedicalNet/MedicalNet_pytorch_files2/pretrain/resnet_18_23dataset.pth'
sets.resume_path = 'resnet_5.pth'
sets.num_workers= 0
sets.model_depth = 10
sets.resnet_shortcut = 'A'
sets.input_D = 5
sets.input_H = 256
sets.batch_size = 16
sets.input_W = 256
sets.fold_index =1
#sets.new_layer_names = nn.Sequential( nn.Dropout(0,5), nn.Linear(10*128*128,6), nn.Sigmoid())
#sets.n_seg_classes = 6
sets.save_folder =r'models_{}_{}_{}_fold_{}'.format('resnet',sets.model_depth,sets.resnet_shortcut,sets.fold_index)
# getting model
if not os.path.exists(sets.save_folder):
        os.makedirs(sets.save_folder)

torch.manual_seed(sets.manual_seed)
model, parameters = generate_model(sets)
"""device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model = model.cuda() if device else model"""
print(model)
sets.phase = 'train'
if sets.no_cuda:
    sets.pin_memory = False
else:
    sets.pin_memory = True

"""load_dict = torch.load('resnet_3d.pt')
model.load_state_dict(load_dict)
model = model.to(device)"""
# optimizer
def get_optimizer(net):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=3e-4, betas=(0.9, 0.999),
                                 eps=1e-08)

    def ajust_lr(optimizer, epoch):
        if epoch < 24:
            lr = 3e-4
        elif epoch < 36:
            lr = 1e-4
        else:
            lr = 1e-5

        for p in optimizer.param_groups:
            p['lr'] = lr
        return lr

    rate = ajust_lr(optimizer, 0)
    return optimizer, ajust_lr

optimizer, ajust_lr = get_optimizer(model)
# train from resume
if sets.resume_path:
    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            print("=> loading checkpoint '{}'".format(sets.resume_path))
            checkpoint = torch.load(sets.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(sets.resume_path, checkpoint['ecpoch']))

# getting data
"""

train_dataset = TreeDCCN_Dataset(csv_file='whole_train3d.csv',path= sets.data_root, transform=albumentations.Compose(
                                               [albumentations.Resize(256, 256),
                                                #albumentations.Normalize(mean=[0.1738, 0.1433, 0.1970], std=[0.3161, 0.2850, 0.3111],
                                                          #max_pixel_value=1.),
                                               ToTensor()]))
train_loader = DataLoader(train_dataset, batch_size=sets.batch_size,
                              shuffle=True, num_workers=sets.num_workers,
                              pin_memory=sets.pin_memory,drop_last=True)


losses, accuracies, hamming_scores, hammings = train(train_loader, valid_loader,model, optimizer,ajust_lr,
          total_epochs=sets.n_epochs,
           save_interval=sets.save_intervals,
           save_folder='models_resnet_10_A_fold_1', sets=sets)


plt.figure()
plt.plot(losses, c='b', label='losses')
plt.plot(hamming_scores, c='g', label='hamming_score')
plt.plot(hammings, c='y', label='hamming_loss')
plt.plot(accuracies, c='r', label='strict_accuracy')
plt.ylabel('Metric')
plt.xlabel('Epoch')
plt.legend()
plt.show()
"""
valid_dataset = FiveSlices_Dataset('one_v3d.csv',sets.data_root, transform= albumentations.Compose(
                                               [albumentations.Resize(256, 256),
                                                albumentations.Normalize(mean=[0.1738], std=[0.3161],
                                                max_pixel_value=1.),
                                               ToTensor()]))

valid_loader = DataLoader(valid_dataset, batch_size=sets.batch_size,
                         shuffle=False, num_workers=sets.num_workers,
                       pin_memory=sets.pin_memory, drop_last=False)
# training
batch_loss = 0
total_t = 0
correct_t = 0
aucs=[]
losses = []
negative_recalls=[]
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
    for batch_id, batch_t in enumerate(valid_loader):
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
        #yhat = outputs_t.round()[:, 1]

        """auc =  roc_auc_score(target_t, outputs_t.round(),average='macro', labels = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural','any'])
        print(f'\nAUC: {auc}')"""

        hamming_lossv = hamming_loss(target_t,outputs_t.round())
        hammings.append(hamming_lossv)
        accuracy = accuracy_score(target_t,outputs_t.round())
        accuracies.append(accuracy)
        hamming_scorev = hamming_score(target_t,outputs_t.round())

        hamming_scores.append(hamming_scorev)
        total_accuracies.append(accuracy)
        total_losses.append(loss_t.item())
        total_hamming_scores.append(hamming_scorev)
        total_hammings.append(hamming_lossv)


        precisions.append(precision_score(target_t,outputs_t.round(),average='macro',zero_division=1))
        f1s.append(f1_score(target_t, outputs_t.round(), average='macro',zero_division=1))
        recalls.append(recall_score(target_t, outputs_t.round(), average='macro',zero_division=1))



        negative_recalls.append(recall_score(my_oposite(target_t), my_oposite(outputs_t.round()), average='macro'))
        """if batch_id % 100 == 0:
            print(f'\nValidation batch: {batch_id}')
            print(f'Validation loss: {loss_t}')
            print(hamming_scorev,hamming_lossv, accuracy)"""

    #auc = np.array(aucs).mean()
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
    print(h_loss,h_score)
    print(precision, f1, recall,negative_recall)

