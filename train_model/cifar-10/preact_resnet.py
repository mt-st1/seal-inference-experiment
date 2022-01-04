# %%
import argparse
import datetime
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchvision.transforms as transforms
# from timm.data.auto_augment import rand_augment_transform
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision import datasets

sys.path.append(os.path.abspath("../utils"))
import activations
from activations import ACTIVATIONS
from plots import plot_and_save_training_history
from torch_hdf5 import save_params_as_hdf5
from torch_json import save_structure_as_json

from models.preact_resnet import PreActResNet10, PreActResNet18

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 200
INPUT_C = 3
INPUT_H = 32
INPUT_W = 32

MODELS = ['preact_resnet10', 'preact_resnet18']

# %%
train_acc, test_acc = torchmetrics.Accuracy(), torchmetrics.Accuracy()
max_test_acc = 0.0
sw = SummaryWriter(log_dir="./histories")

running_loss_history = []
running_acc_history = []
val_running_loss_history = []
val_running_acc_history = []

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, choices=MODELS)
parser.add_argument('--act', required=True, choices=ACTIVATIONS)

args = parser.parse_args()
model_str = args.model
act_str = args.act

if model_str == 'preact_resnet10':
    build_model_fn = PreActResNet10
elif model_str == 'preact_resnet18':
    build_model_fn = PreActResNet18

if act_str == 'relu':
    activation = nn.ReLU
elif act_str == 'swish':
    activation = activations.Swish
elif act_str == 'mish':
    activation = nn.Mish
elif act_str == 'square':
    activation = activations.Square
# ReLU approx.
elif act_str == 'relu_rg4_deg2':
    activation = activations.ReluRg4Deg2
elif act_str == 'relu_rg5_deg2':
    activation = activations.ReluRg5Deg2
elif act_str == 'relu_rg6_deg2':
    activation = activations.ReluRg6Deg2
elif act_str == 'relu_rg7_deg2':
    activation = activations.ReluRg7Deg2
elif act_str == 'relu_rg4_deg4':
    activation = activations.ReluRg4Deg4
elif act_str == 'relu_rg5_deg4':
    activation = activations.ReluRg5Deg4
elif act_str == 'relu_rg6_deg4':
    activation = activations.ReluRg6Deg4
elif act_str == 'relu_rg7_deg4':
    activation = activations.ReluRg7Deg4
# Swish approx.
elif act_str == 'swish_rg4_deg2':
    activation = activations.SwishRg4Deg2
elif act_str == 'swish_rg5_deg2':
    activation = activations.SwishRg5Deg2
elif act_str == 'swish_rg6_deg2':
    activation = activations.SwishRg6Deg2
elif act_str == 'swish_rg7_deg2':
    activation = activations.SwishRg7Deg2
elif act_str == 'swish_rg4_deg4':
    activation = activations.SwishRg4Deg4
elif act_str == 'swish_rg5_deg4':
    activation = activations.SwishRg5Deg4
elif act_str == 'swish_rg6_deg4':
    activation = activations.SwishRg6Deg4
elif act_str == 'swish_rg7_deg4':
    activation = activations.SwishRg7Deg4
# Mish approx.
elif act_str == 'mish_rg4_deg2':
    activation = activations.MishRg4Deg2
elif act_str == 'mish_rg5_deg2':
    activation = activations.MishRg5Deg2
elif act_str == 'mish_rg6_deg2':
    activation = activations.MishRg6Deg2
elif act_str == 'mish_rg7_deg2':
    activation = activations.MishRg7Deg2
elif act_str == 'mish_rg4_deg4':
    activation = activations.MishRg4Deg4
elif act_str == 'mish_rg5_deg4':
    activation = activations.MishRg5Deg4
elif act_str == 'mish_rg6_deg4':
    activation = activations.MishRg6Deg4
elif act_str == 'mish_rg7_deg4':
    activation = activations.MishRg7Deg4

# %%
now = datetime.datetime.now()
md_str = now.strftime('%m%d')
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_FILE_NAME = f"{model_str}-{act_str}"
SAVE_MODEL_DIR_NAME = "saved_models"
BEST_MODEL_STATE_DICT_PATH = f"{CUR_DIR}/{SAVE_MODEL_DIR_NAME}/{BASE_FILE_NAME}-{md_str}-best.pt"


# %%
def train_one_epoch(epoch, model, train_loader, val_loader, loss_func, optimizer):
    model.train()

    total_step = len(train_loader)
    running_loss = 0.0
    running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # Clear the gradients of all parameters
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        batch_out = model(inputs)
        loss = loss_func(batch_out, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(batch_out, -1)
        running_loss += loss.cpu().item()
        running_corrects += torch.sum(preds == labels.data).cpu().item()
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
                val_loss = loss_func(val_out, val_labels)

                val_preds = torch.argmax(val_out, dim=-1)
                val_running_loss += val_loss.cpu().item()
                val_running_corrects += torch.sum(val_preds == val_labels.data).cpu().item()
                test_acc(val_preds.cpu(), val_labels.cpu())

        epoch_loss = running_loss / len(train_loader)  # loss per epoch
        epoch_acc = running_corrects / len(train_loader.dataset)  # accuracy per epoch
        running_loss_history.append(epoch_loss)
        running_acc_history.append(epoch_acc)

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = val_running_corrects / len(val_loader.dataset)
        val_running_loss_history.append(val_epoch_loss)
        val_running_acc_history.append(val_epoch_acc)

        print(f'training loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}')
        print(f'validation loss: {val_epoch_loss:.4f}, validation acc: {val_epoch_acc:.4f}')
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
def main():
    print(f"Device: {DEVICE}\n")

    # train_transform = transforms.Compose([rand_augment_transform(config_str='rand-m8-n4-mstd0.5',
    #                                                              hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}),
    #                                       transforms.ToTensor(),
    #                                       transforms.Normalize(mean=(0.5,), std=(0.5,))])
    # train_transform = transforms.Compose([rand_augment_transform(config_str='rand-m8-n4-mstd0.5', hparams={'translate_const': 117}),
    #                                       transforms.ToTensor(),
    #                                       transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    #                                     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    #                               transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])

    # Download CIFAR10 dataset
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

    # Define data loaders
    loaders = {
        "train": DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        "test": DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }

    model = build_model_fn(activation)
    summary(model, input_size=(1, INPUT_C, INPUT_H, INPUT_W))
    print()

    # Define loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())
    scheduler = CosineLRScheduler(optimizer, t_initial=EPOCHS, lr_min=1e-4, warmup_t=20, warmup_lr_init=1e-4, warmup_prefix=True)

    # Train model
    model = model.to(DEVICE)
    for epoch in range(EPOCHS):
        print(f"Epoch [{epoch+1:3d}/{EPOCHS:3d}]")
        train_acc.reset(), test_acc.reset()
        train_one_epoch(epoch, model, loaders["train"], loaders["test"], loss_func, optimizer)
        scheduler.step(epoch+1)
        print()

    print(f"Finished training! (Best accuracy: {max_test_acc})")
    print()

    # Test model
    best_model = build_model_fn(activation)
    best_model.load_state_dict(torch.load(BEST_MODEL_STATE_DICT_PATH, map_location=torch.device(DEVICE)))
    best_model = best_model.to(DEVICE)
    start = time.perf_counter()
    test_accuracy = test(best_model, loaders["test"])
    end = time.perf_counter()
    print(f"Test time using model state dict: {end - start} [sec]\n")

    now = datetime.datetime.now()
    md_hm_str = now.strftime('%m%d_%H%M')
    test_acc_str = f'{test_accuracy:.2f}'
    common_name = f'{BASE_FILE_NAME}-{test_acc_str}_{EPOCHS}epoch-{md_hm_str}'
    MODEL_STRUCTURE_PATH = f"{CUR_DIR}/{SAVE_MODEL_DIR_NAME}/{BASE_FILE_NAME}-structure.json"
    MODEL_PARAMS_PATH = f"{CUR_DIR}/{SAVE_MODEL_DIR_NAME}/{common_name}-params.h5"
    HISTORY_GRAPH_PATH = f"{CUR_DIR}/graphs/{common_name}_history.png"

    # Save model structure in JSON and params in hdf5
    save_structure_as_json(model, MODEL_STRUCTURE_PATH)
    best_model = best_model.to('cpu')
    save_params_as_hdf5(best_model, MODEL_PARAMS_PATH)

    # Plot history graph (accuracy & loss) and save
    plot_and_save_training_history(running_loss_history,
                                   val_running_loss_history,
                                   running_acc_history,
                                   val_running_acc_history,
                                   HISTORY_GRAPH_PATH)


# %%
if __name__ == '__main__':
    main()
