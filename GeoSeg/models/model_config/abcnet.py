from torch.utils.data import DataLoader
from GeoSeg.models.tools.losses.joint_loss import *
from GeoSeg.models.tools.losses.soft_ce import *
from GeoSeg.models.tools.losses.dice import *
from GeoSeg.models.tools.losses.functional import *
from GeoSeg.data_processing.Urban_dataset import *
from GeoSeg.models.ABCNet import ABCNet
from catalyst.contrib.nn import Lookahead
from catalyst import utils
from torchvision import transforms

# training hparam
max_epoch = 200
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 8
lr = 1e-4
weight_decay = 5e-4
backbone_lr = 1e-5
backbone_weight_decay = 5e-4
num_classes = len(CLASSES)
classes = CLASSES
gpus = [0]

weights_name = "abcnet-512-crop-ms-e200"
weights_path = "GeoSeg/models/ckpt/"
test_weights_name = "abcnet-512-crop-ms-e200"
log_name = weights_name
monitor = 'val_OA'
monitor_mode = 'max'
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
weightgpus = [0]  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
net = ABCNet(n_classes=num_classes)

# define the loss
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

use_aux_loss = False

# Datasets
train_data_root = 'GeoSeg/data/Urban/train' 
val_data_root = 'GeoSeg/data/Urban/val' 

train_dataset = UrbanDataset(data_root=val_data_root,
                             mode='test',
                             augmentation=train_aug)

val_dataset = UrbanDataset(data_root=val_data_root,
                             mode='val',
                             augmentation=val_aug)

test_dataset = UrbanDataset(data_root=val_data_root,
                             mode='test',
                             augmentation=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
