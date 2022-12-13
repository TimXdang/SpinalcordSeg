# Train the model
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from models.models import UNet2DRemake, UNet3D
import segmentation_models_pytorch.utils as smpu
from models.dataset_classes import *
from utils import *

# configurations which can be replaced by config file later on
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = UNet2DRemake().model
model = UNet3D(in_channels=1, out_channels=1)
loss = smpu.losses.DiceLoss()
metrics = [smpu.metrics.IoU(threshold=0.5), ]

num_epochs = 30
batch_size = 16
# define after how many epochs the learning rate shall be changed
epochs_to_decay = 50
reduced_lr = 1e-5

optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.002), ])

# set up training and validation
# train_epoch = smpu.train.TrainEpoch(model=model, loss=loss, metrics=metrics, optimizer=optimizer,
#                                     device=device, verbose=True)
# valid_epoch = smpu.train.ValidEpoch(model=model, loss=loss, metrics=metrics, device=device, verbose=True)

train_epoch = TrainEpoch(model=model, loss=loss, metrics=metrics, optimizer=optimizer, device=device, verbose=True,
                         #unet2d=True
                         )
valid_epoch = ValidEpoch(model=model, loss=loss, metrics=metrics, device=device, verbose=True,
                         #unet2d=True
                         )

# set up k-fold cross validation
k = 5
splits = KFold(n_splits=k, shuffle=True, random_state=42)
foldperf = {}

# dataset, _ = divide_data()
new_set = Experiment4('dataset/Experiment4/sub-02', (0, 2, 3, 225))
dataset = new_set.create_dataset(augmentation=True)

# start training
for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
    print('\nFold {}'.format(fold + 1))

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    # create logs
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print('\nEpoch: {}/{}'.format(epoch + 1, num_epochs))

        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)

        train_loss = train_logs['dice_loss']
        train_acc = train_logs['iou_score']
        val_loss = valid_logs['dice_loss']
        val_acc = valid_logs['iou_score']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # if fold * num_epochs + epoch + 1 == epochs_to_decay:
        #     optimizer.param_groups[0]['lr'] = reduced_lr
        #     print('Decrease decoder learning rate to {}!'.format(reduced_lr))

    foldperf['fold{}'.format(fold + 1)] = history

# save the model
torch.save(model, 'output/k_cross.pth')

# average loss and accuracy per fold
tl_f, vl_f, ta_f, va_f = [], [], [], []
# loss and accuracy of all folds concatenated
tl_over_f, vl_over_f, ta_over_f, va_over_f = [], [], [], []

for f in foldperf:
    tl_f.append(np.mean(foldperf['{}'.format(f)]['train_loss']))
    vl_f.append(np.mean(foldperf['{}'.format(f)]['val_loss']))
    ta_f.append(np.mean(foldperf['{}'.format(f)]['train_acc']))
    va_f.append(np.mean(foldperf['{}'.format(f)]['val_acc']))
    tl_over_f.extend(foldperf['{}'.format(f)]['train_loss'])
    vl_over_f.extend(foldperf['{}'.format(f)]['val_loss'])
    ta_over_f.extend(foldperf['{}'.format(f)]['train_acc'])
    va_over_f.extend(foldperf['{}'.format(f)]['val_acc'])

print('Performance of {} fold cross validation'.format(k))
print('Average Training Loss: {:.3f} \t Average Val Loss: {:.3f} \t Average Training Acc: {:.2f}'
      '\t Average Val Acc: {:.2f}'.format(np.mean(tl_f), np.mean(vl_f), np.mean(ta_f), np.mean(va_f)))

# create plots
x_axis = np.arange(len(tl_over_f)) + 1
plt.plot(x_axis, tl_over_f, label='Train')
plt.plot(x_axis, vl_over_f, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Dice Loss')
plt.title('U-Net Loss Curves')
plt.legend()
plt.savefig('output/loss_curves')

plt.clf()

plt.plot(x_axis, ta_over_f, label='Train')
plt.plot(x_axis, va_over_f, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Jaccard Index')
plt.title('U-Net Accuracy')
plt.legend()
plt.savefig('output/acc_curves')
