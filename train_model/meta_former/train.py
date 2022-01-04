# %%
import datetime
import os
import sys

import torch
import torchmetrics
# import torch.nn as nn
import torchvision.transforms as transforms
from timm.data.auto_augment import rand_augment_transform
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler
# from torch.autograd import Variable
from torch.optim import AdamW
# from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision import datasets
from torchvision.transforms.functional import InterpolationMode

from models.pool_former import PoolFormer

sys.path.append(os.path.abspath("../utils"))
from plots import plot_and_save_training_history
from torch_hdf5 import save_params_as_hdf5
from torch_json import save_structure_as_json

# %%
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# BATCH_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 200
INPUT_H = 128
INPUT_W = 128

now = datetime.datetime.now()
ymd_str = now.strftime('%y%m%d')
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
SAVE_MODEL_DIR_NAME = "saved_models"
BEST_MODEL_STATE_DICT_PATH = f"{CUR_DIR}/{SAVE_MODEL_DIR_NAME}/{BASE_FILE_NAME}-{ymd_str}-best.pt"

# %%
train_acc, test_acc = torchmetrics.Accuracy(), torchmetrics.Accuracy()
max_test_acc = 0.0
sw = SummaryWriter(log_dir="./logs")

# %%
# class WarmupScheduler(_LRScheduler):
#     def __init__(self, optimizer, lr, min_lr_ratio, total_steps, warmup):
#         self.lr = lr
#         self.min_lr_ratio = min_lr_ratio
#         self.total_steps = total_steps
#         self.warmup = warmup
#         super(WarmupScheduler, self).__init__(optimizer)

#     def get_lr(self):
#         step = super(WarmupScheduler, self)
#         step_l = step-self.warmup


# %%
running_loss_history = []
running_acc_history = []
val_running_loss_history = []
val_running_acc_history = []


# %%
def train_one_epoch(epoch, model, train_loader, val_loader, loss_func, optimizer):
    # steps_per_epoch = 50000 // BATCH_SIZE
    model.train()

    total_step = len(train_loader)
    running_loss = 0.0
    running_corrects = 0.0
    # val_running_loss = 0.0
    # val_running_corrects = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        inputs, labels_mixup = mixup_fn(inputs, labels)
        # batch_x = Variable(inputs)
        # batch_y = Variable(labels)

        # Clear the gradients of all parameters
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        batch_out = model(inputs)
        loss = loss_func(batch_out, labels_mixup)
        loss.backward()
        optimizer.step()

        # _, preds = torch.max(batch_out, 1)
        preds = torch.argmax(batch_out, -1)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)
        train_acc(preds.cpu(), labels.cpu())

        # Print statistics (per 100 iterations and end of epoch)
        if (i + 1) % 100 == 0 or (i + 1) == total_step:
            print(f"Step [{i+1:3d}/{total_step:3d}] -> Loss: {loss.item():.4f}")

    else:
        model.eval()
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(DEVICE)
                val_labels = val_labels.to(DEVICE)
                val_out = model(val_inputs)
                # val_loss = loss_func(val_out, val_labels)

                # _, val_preds = torch.max(val_out, 1)
                val_preds = torch.argmax(val_out, dim=-1)
                # val_running_loss += val_loss.item()
                # val_running_corrects += torch.sum(val_preds == val_labels.data)
                test_acc(val_preds.cpu(), val_labels.cpu())

        epoch_loss = running_loss / len(train_loader)  # loss per epoch
        epoch_acc = running_corrects.float() / len(train_loader)  # accuracy per epoch
        running_loss_history.append(epoch_loss)
        running_acc_history.append(epoch_acc)

        # val_epoch_loss = val_running_loss / len(val_loader)
        # val_epoch_acc = val_running_corrects / len(val_loader)
        # val_running_loss_history.append(val_epoch_loss)
        # val_running_acc_history.append(val_epoch_acc)

        print(f'training loss: {epoch_loss:.4f}, acc: {epoch_acc.item():.4f}')
        # print(f'validation loss: {val_epoch_loss:.4f}, validation acc: {val_epoch_acc.item():.4f}')
        print(f'TrainAcc: {train_acc.compute()}, TestAcc: {test_acc.compute()}')
        global max_test_acc
        if max_test_acc < test_acc.compute():
            print(f'Test acc improved from {max_test_acc} to {test_acc.compute()}')
            torch.save(model.state_dict(), BEST_MODEL_STATE_DICT_PATH)
            print('Model saved.')
            max_test_acc = test_acc.compute()
        sw.add_scalar('Train Accuracy', train_acc.compute(), epoch+1)
        sw.add_scalar('Test Accuracy', test_acc.compute(), epoch+1)


# %%
def test(model, data_loader):
    model.eval()  # Inference mode

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)
            preds = torch.max(outputs, dim=1)[1]
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100 * float(correct / total)
    print(f"Accuracy: {acc:.2f}")
    return acc


# %%
# train_transform = transforms.Compose([transforms.Resize((INPUT_H, INPUT_W), interpolation=InterpolationMode.BICUBIC),
#                                       transforms.RandomHorizontalFlip(),
#                                       transforms.RandomRotation(10),
#                                       transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
#                                       transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_transform = transforms.Compose([transforms.Resize((INPUT_H, INPUT_W), interpolation=InterpolationMode.BICUBIC),
                                      # RandAugment
                                      rand_augment_transform(config_str='rand-m9-mstd0.5',
                                                             hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose([transforms.Resize((INPUT_H, INPUT_W), interpolation=InterpolationMode.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = datasets.CIFAR10(root="../cifar-10/data", train=True, download=True, transform=train_transform)
test_data = datasets.CIFAR10(root="../cifar-10/data", train=False, download=True, transform=transform)

print("<Train data>")
print(train_data)
print()
print(f"Train data images: {train_data.data.shape}")
print(f"Train data labels: {len(train_data.targets)}\n")
print("<Test data>")
print(test_data)
print()
print(f"Test data images: {test_data.data.shape}")
print(f"Test data labels: {len(test_data.targets)}\n")

# %%
loaders = {
    "train": DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    "test": DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
}

# %%
# model = PoolFormer(token_mixier_type='pooling', in_patch_size=3, in_stride=1, in_pad=1, width=0.5, num_classes=10)
model = PoolFormer(token_mixier_type='pooling', in_patch_size=3, in_stride=1, in_pad=1, width=1., num_classes=10)

# %%
summary(model, input_size=(1, 3, INPUT_H, INPUT_W))

# %%
# loss_func = nn.CrossEntropyLoss()
loss_func = SoftTargetCrossEntropy()
optimizer = AdamW(model.parameters(), weight_decay=0.05)
scheduler = CosineLRScheduler(optimizer, t_initial=EPOCHS, lr_min=1e-4, warmup_t=20, warmup_lr_init=1e-4, warmup_prefix=True)

# %%
# mixup + cutmix
mixup_args = {
    'mixup_alpha': 0.,
    'cutmix_alpha': 1.0,
    'cutmix_minmax': None,
    'prob': 1.0,
    'switch_prob': 0.,
    'mode': 'batch',
    'label_smoothing': 0,
    'num_classes': 10
}
mixup_fn = Mixup(**mixup_args)

# %%
# train_acc, test_acc = torchmetrics.Accuracy(), torchmetrics.Accuracy()
# max_test_acc = 0.0
# sw = SummaryWriter(log_dir="./logs")

# %%
# Train model
model = model.to(DEVICE)
for epoch in range(EPOCHS):
    print(f"Epoch [{epoch+1:3d}/{EPOCHS:3d}]")
    train_acc.reset(), test_acc.reset()
    train_one_epoch(epoch, model, loaders["train"], loaders["test"], loss_func, optimizer)
    # lr_scheduler.step()
    scheduler.step(epoch+1)
    print()

print("Finished training!")
print()


# %%
# Test model
test_acc = test(model, loaders["test"])
print()

# %%
now = datetime.datetime.now()
ymd_hm_str = now.strftime('%y%m%d_%H%M')
test_acc_str = f'{test_acc:.2f}'
common_name = f'{BASE_FILE_NAME}-{test_acc_str}-{ymd_hm_str}'
MODEL_STRUCTURE_PATH = f"{CUR_DIR}/{SAVE_MODEL_DIR_NAME}/{common_name}-structure.json"
MODEL_PARAMS_PATH = f"{CUR_DIR}/{SAVE_MODEL_DIR_NAME}/{common_name}-params.h5"
HISTORY_GRAPH_PATH = f"{CUR_DIR}/graphs/{common_name}_history.png"

# %%
# Save model structure as JSON
save_structure_as_json(model, MODEL_STRUCTURE_PATH)

# %%
# Save model parameters as hdf5
model = model.to('cpu')
save_params_as_hdf5(model, MODEL_PARAMS_PATH)

# %%
# Plot history graph (accuracy & loss) and save
plot_and_save_training_history(running_loss_history,
                               val_running_loss_history,
                               running_acc_history,
                               val_running_acc_history,
                               HISTORY_GRAPH_PATH)
