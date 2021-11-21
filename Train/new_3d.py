import torch
from MedicalNet.setting import parse_opts
from MedicalNet.model import generate_model
import os
import matplotlib.pyplot as plt
from Datasets.ThreeDDataset import ThreeDCCNDataset
import pickle
from sklearn.metrics import hamming_loss, accuracy_score
import albumentations
from torch.utils.data import DataLoader
import numpy as np
from albumentations.pytorch import ToTensor
from MedicalNet.utils.logger import log

path = 'MedicalNet/MedicalNet_pytorch_files2/pretrain/resnet_18.pth'


def hamming_score(y_true, y_pred):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])

        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / \
                    float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)

    return np.mean(acc_list)


def train(data_loader, data_loader_valid, model, optimizer, total_epochs, save_interval, save_folder, sets):
    # settings
    total_accuracies = []
    total_losses = []
    valid_loss_min = np.Inf
    total_hamming_scores = []
    total_hammings = []
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    criterion = torch.nn.BCELoss()

    print("Current setting is:")
    print(sets)
    print("\n\n")
    if not sets.no_cuda:
        criterion = criterion.cuda()

    model.train()
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))

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

            log.info(
                'Batch: {}-{} ({}), loss = {:.3f}'
                .format(epoch, batch_id, batch_id_sp / 10, loss.item()))

            if not sets.ci_test:
                # save model
                if batch_id == 0 and batch_id_sp != 0:
                    # if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    model_save_path = '{}_epoch_{}_batch_{}.pth'.format(save_folder, epoch, batch_id)
                    print(model_save_path)
                    print(model_save_path)
                    if not os.path.exists(model_save_path):
                        os.makedirs(model_save_path)

                    log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id))
                    torch.save({
                        'ecpoch': epoch,
                        'batch_id': batch_id,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        model_save_path)

        losses = []
        accuracies = []
        hammings = []
        hamming_scores = []
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

            accuracy = np.array(accuracies).mean()
            batch_loss = np.array(losses).mean()
            hamming_scorev = np.array(hamming_scores).mean()

            print(f'validation loss: {batch_loss:.4f}, validation hamming loss: {hamming_scorev :.4f}\n')

            network_learned = batch_loss < valid_loss_min
            print(f'validation loss: {batch_loss:.4f}, validation acc: {accuracy:.4f}\n')

            if network_learned:
                valid_loss_min = batch_loss
                torch.save(model.state_dict(), 'resnet_3d_1.pt')
                print('Improvement-Detected, save-model')
        model.train()

    print('Finished training')
    return total_losses, total_accuracies, total_hamming_scores, total_hammings


def my_oposite(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j] == 0:
                x[i][j] = 1
            else:
                x[i][j] = 0

    return x


sets = parse_opts()
sets.gpu_id = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]

sets.n_epochs = 5
sets.no_cuda = False
sets.data_root = '../rsna-intracranial-hemorrhage-detection/stage_2_train/'
sets.pretrain_path = 'MedicalNet/MedicalNet_pytorch_files2/pretrain/resnet_18_23dataset.pth'
sets.resume_path = 'resnet_5.pth'
sets.num_workers = 0
sets.model_depth = 10
sets.resnet_shortcut = 'A'
sets.input_D = 5
sets.input_H = 256
sets.batch_size = 16
sets.input_W = 256
sets.fold_index = 1
# sets.new_layer_names = nn.Sequential( nn.Dropout(0,5), nn.Linear(10*128*128,6), nn.Sigmoid())
# sets.n_seg_classes = 6
sets.save_folder = r'models_{}_{}_{}_fold_{}'.format('resnet', sets.model_depth, sets.resnet_shortcut, sets.fold_index)
# getting model
if not os.path.exists(sets.save_folder):
    os.makedirs(sets.save_folder)

torch.manual_seed(sets.manual_seed)
model, parameters = generate_model(sets)

print(model)
sets.phase = 'train'
if sets.no_cuda:
    sets.pin_memory = False
else:
    sets.pin_memory = True


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

train_dataset = ThreeDCCNDataset(csv_file='../build/whole_train3d.csv', path=sets.data_root,
                                 transform=albumentations.Compose(
                                     [albumentations.Resize(256, 256),
                                      # albumentations.Normalize(mean=[0.1738, 0.1433, 0.1970], std=[0.3161, 0.2850, 0.3111],
                                      # max_pixel_value=1.),
                                      ToTensor()]))
train_loader = DataLoader(train_dataset, batch_size=sets.batch_size,
                          shuffle=True, num_workers=sets.num_workers,
                          pin_memory=sets.pin_memory, drop_last=True)

valid_dataset = ThreeDCCNDataset('../build/whole_train3d.csv', path=sets.data_root, transform=albumentations.Compose(
    [albumentations.Resize(256, 256),
     albumentations.Normalize(mean=[0.1738], std=[0.3161],
                              max_pixel_value=1.),
     ToTensor()]))

valid_loader = DataLoader(valid_dataset, batch_size=sets.batch_size,
                          shuffle=False, num_workers=sets.num_workers,
                          pin_memory=sets.pin_memory, drop_last=False)

losses, accuracies, hamming_scores, hammings = train(train_loader, valid_loader, model, optimizer,
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
