# Train the model
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from models.models import UNet2DRemake, UNet3D
import segmentation_models_pytorch.utils as smpu
from models.dataset_classes import *
from utils import *
from loss import *
from metric import *

# configurations which can be replaced by config file later on
device = torch.device(#'cpu'
    'cuda:0' if torch.cuda.is_available() else 'cpu'
                      )
# model = UNet2DRemake().model
model = UNet3D(in_channels=1, out_channels=1)
loss = smpu.losses.DiceLoss()
loss = VolDiceLoss()
metrics = [smpu.metrics.IoU(threshold=0.5), ]
metrics = [JaccardIndex(), ]

num_epochs = 5
batch_size = 8
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

    if fold == 1:
        optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.001), ])
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    if fold == 2:
        optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.003), ])
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    if fold == 3:
        optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.002), ])
        train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=4, sampler=val_sampler)
    if fold == 4:
        loss = CrossEntropyDice(alpha=0.5, beta=0.5)
        optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.002), ])
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    model = UNet3D(in_channels=1, out_channels=1)

    train_epoch = TrainEpoch(model=model, loss=loss, metrics=metrics, optimizer=optimizer, device=device, verbose=True,
                             # unet2d=True
                             )
    valid_epoch = ValidEpoch(model=model, loss=loss, metrics=metrics, device=device, verbose=True,
                             # unet2d=True
                             )

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

    # create plots
    x_axis = np.arange(num_epochs) + 1
    plt.plot(x_axis, history['train_loss'], label='Train')
    plt.plot(x_axis, history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.title('U-Net Loss Curves')
    plt.legend()
    plt.savefig(f'output/loss_curves_{fold}')

    plt.clf()

    plt.plot(x_axis, history['train_acc'], label='Train')
    plt.plot(x_axis, history['val_acc'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Jaccard Index')
    plt.title('U-Net Accuracy')
    plt.legend()
    plt.savefig(f'output/acc_curves_{fold}')

    plt.clf()
