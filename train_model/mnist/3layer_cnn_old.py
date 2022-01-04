# %%
import argparse
import datetime
import os
import sys
import time

import torch
import torch.nn as nn
import torchmetrics
import torchvision.transforms as transforms
# from timm.data.auto_augment import rand_augment_transform
from torch.optim import Adam
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

# %%
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 150
INPUT_C = 1
INPUT_H = 28
INPUT_W = 28

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
parser.add_argument('--bn', action='store_true')
parser.add_argument('--act', required=True, choices=ACTIVATIONS)

args = parser.parse_args()
bn_enabled = args.bn
act_str = args.act

if act_str == 'relu':
    activation = nn.ReLU
elif act_str == 'swish':
    activation = activations.Swish
elif act_str == 'mish':
    activation = nn.Mish
elif act_str == 'square':
    activation = activations.Square
elif act_str == 'relu_rg4_deg4':
    activation = activations.ReluRg4Deg4
elif act_str == 'relu_rg6_deg4':
    activation = activations.ReluRg6Deg4
elif act_str == 'relu_rg4_deg2':
    activation = activations.ReluRg4Deg2
elif act_str == 'relu_rg6_deg2':
    activation = activations.ReluRg6Deg2
elif act_str == 'swish_rg4_deg4':
    activation = activations.SwishRg4Deg4
elif act_str == 'swish_rg6_deg4':
    activation = activations.SwishRg6Deg4
elif act_str == 'swish_rg4_deg2':
    activation = activations.SwishRg4Deg2
elif act_str == 'swish_rg6_deg2':
    activation = activations.SwishRg6Deg2
elif act_str == 'mish_rg4_deg4':
    activation = activations.MishRg4Deg4
elif act_str == 'mish_rg6_deg4':
    activation = activations.MishRg6Deg4
elif act_str == 'mish_rg4_deg2':
    activation = activations.MishRg4Deg2
elif act_str == 'mish_rg6_deg2':
    activation = activations.MishRg6Deg2


# %%
now = datetime.datetime.now()
md_str = now.strftime('%m%d')
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if bn_enabled:
    BASE_FILE_NAME = f"{os.path.splitext(os.path.basename(__file__))[0]}-{act_str}-BN"
else:
    BASE_FILE_NAME = f"{os.path.splitext(os.path.basename(__file__))[0]}-{act_str}"
SAVE_MODEL_DIR_NAME = "saved_models"
BEST_MODEL_STATE_DICT_PATH = f"{CUR_DIR}/{SAVE_MODEL_DIR_NAME}/{BASE_FILE_NAME}-{md_str}-best.pt"


# %%
class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=2)
        self.act1 = activation()
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=50, kernel_size=5, stride=2)
        self.act2 = activation()
        self.fc = nn.Linear(50 * 4 * 4, 10)
        if bn_enabled:
            self.norm1 = nn.BatchNorm2d(5)
            self.norm2 = nn.BatchNorm2d(50)

    def forward(self, x):
        x = self.conv1(x)
        if bn_enabled:
            x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        if bn_enabled:
            x = self.norm2(x)
        x = self.act2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


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

        preds = torch.argmax(batch_out, dim=-1)
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
        epoch_acc = running_corrects / len(train_loader)  # accuracy per epoch
        running_loss_history.append(epoch_loss)
        running_acc_history.append(epoch_acc)

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = val_running_corrects / len(val_loader)
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


def main():
    print(f"Device: {DEVICE}\n")

    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(10),
                                          transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=10, scale=(0.8, 1.2)),
                                          transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.5,), std=(0.5,))])
    # train_transform = transforms.Compose([rand_augment_transform(config_str='rand-m9-mstd0.5',
    #                                                              hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}),
    #                                       transforms.ToTensor(),
    #                                       transforms.Normalize((0.5,), (0.5,))])
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,), std=(0.5,))])

    # Download MNIST dataset
    train_data = datasets.MNIST(root="./data", train=True, transform=train_transform, download=True)
    test_data = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

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

    model = MnistCNN()
    summary(model, input_size=(1, INPUT_C, INPUT_H, INPUT_W))
    print()

    # Define loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    # Train model
    model = model.to(DEVICE)
    for epoch in range(EPOCHS):
        print(f"Epoch [{epoch+1:3d}/{EPOCHS:3d}]")
        train_acc.reset(), test_acc.reset()
        train_one_epoch(epoch, model, loaders["train"], loaders["test"], loss_func, optimizer)
        print()

    print(f"Finished training! (Best accuracy: {max_test_acc})")
    print()

    # Test model
    best_model = MnistCNN()
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


if __name__ == '__main__':
    main()
