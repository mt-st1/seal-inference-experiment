# %%
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
# from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import datasets

sys.path.append(os.path.abspath("../utils"))
from torch_hdf5 import save_params_as_hdf5
from torch_json import save_structure_as_json

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 20
# IMG_HIGHT, IMG_WIDTH = 28, 28

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
SAVE_MODEL_DIR_NAME = "saved_models"
MODEL_STRUCTURE_PATH = f"{CUR_DIR}/{SAVE_MODEL_DIR_NAME}/{BASE_FILE_NAME}-structure.json"
MODEL_PARAMS_PATH = f"{CUR_DIR}/{SAVE_MODEL_DIR_NAME}/{BASE_FILE_NAME}-params.h5"
MODEL_STATE_DICT_PATH = (
    f"{CUR_DIR}/{SAVE_MODEL_DIR_NAME}/{BASE_FILE_NAME}-state_dict.pt"
)
MODEL_TORCH_SCRIPT_PATH = f"{CUR_DIR}/{SAVE_MODEL_DIR_NAME}/{BASE_FILE_NAME}-trace.pt"


# %%
class Square(nn.Module):
    def __init__(self):
        super(Square, self).__init__()

    def forward(self, x):
        return x * x


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=5, kernel_size=5, stride=2
            ),
            Square(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=5, out_channels=50, kernel_size=5, stride=2
            ),
            Square()
        )
        self.fc = nn.Linear(50*4*4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # flatten the output of conv2 to (batch_size, 32x7x7)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# %%
# Train 1 epoch
def train(model, data_loader, loss_func, optimizer):
    model.train()  # Training mode

    total_step = len(data_loader)

    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        batch_x = Variable(inputs)
        batch_y = Variable(labels)

        # Clear the gradients of all parameters
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        batch_out = model(batch_x)
        loss = loss_func(batch_out, batch_y)
        loss.backward()
        optimizer.step()

        # Print statistics (per 100 iterations and end of epoch)
        if (i + 1) % 100 == 0 or (i + 1) == total_step:
            print(f"Step [{i+1:3d}/{total_step:3d}] -> Loss: {loss.item():.4f}")


# Evaluate on test data
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

    print(f"Accuracy: {100 * float(correct / total):.2f}")


# %%
print(f"Device: {DEVICE}\n")

# %%
# Download MNIST dataset
train_data = datasets.MNIST(
    root="data", train=True, transform=transforms.ToTensor(), download=True
)
test_data = datasets.MNIST(
    root="data", train=False, transform=transforms.ToTensor(), download=True
)
print("<Train data>")
print(train_data)
print()
print(f"Train data images: {train_data.data.size()}")
print(f"Train data labels: {train_data.targets.size()}\n")
print("<Test data>")
print(test_data)
print()
print(f"Test data images: {test_data.data.size()}")
print(f"Test data labels: {test_data.targets.size()}\n")

# %%
# Define data loaders
loaders = {
    "train": DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    "test": DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
}

# %%
model = MnistCNN()
print(model)
print()
params = list(model.parameters())
print(f"len(params): {len(params)}")
print(f"conv1's weight: {params[0].size()}")
print(f"conv1's bias: {params[1].size()}")
print(f"conv2's weight: {params[2].size()}")
print(f"conv2's bias: {params[3].size()}")
print(f"fc's weight: {params[4].size()}")
print(f"fc's bias: {params[5].size()}")
print()

# %%
summary(model, input_size=(BATCH_SIZE, 1, 28, 28))

# %%
print(model._get_name())
print(model)

# %%
for name, param in model.named_parameters():
    print(f"name: {name}")
    print(f"\tparam: {param.size()}")


# %%
def print_linear_module(module, indent_level):
    indents = "\t" * indent_level
    if (isinstance(module, nn.Conv2d)):
        params = list(module.parameters())
        weight = params[0]
        bias = params[1]
        print(f"{indents}weight: {weight.size()}")
        print(f"{indents}bias: {bias.size()}")
        print(f"{indents}filters: {weight.size()[0]}")
        print(f"{indents}kernel_size: {[weight.size()[2], weight.size()[3]]}")
        print(f"{indents}strides: {module.stride}")
        print(f"{indents}padding: {module.padding}")
    elif (isinstance(module, nn.AvgPool2d)):
        print(f"{indents}kernel_size: {module.kernel_size}")
        print(f"{indents}strides: {module.stride}")
        print(f"{indents}padding: {module.padding}")
    elif (isinstance(module, nn.BatchNorm2d)):
        params = list(module.parameters())
        weight = params[0]
        bias = params[1]
        print(f"{indents}weight: {weight.size()}")
        print(f"{indents}bias: {bias.size()}")
        print(f"{indents}eps: {module.eps}")
    elif (isinstance(module, nn.Linear)):
        params = list(module.parameters())
        weight = params[0]
        bias = params[1]
        print(f"{indents}weight: {weight.size()}")
        print(f"{indents}bias: {bias.size()}")
        print(f"{indents}units: {weight.size()[0]}")


def print_module_info(module, indent_level=0):
    indents = "\t" * indent_level
    print(f"{indents}name: {module._get_name()}, type: {type(module)}")
    if isinstance(module, (nn.Conv2d, nn.AvgPool2d, nn.BatchNorm2d, nn.Linear)):
        print_linear_module(module, indent_level)
    else:
        print(f"{indents}not instance of Conv2D, AvgPool2D, BatchNorm2D, Linear")
        for name, child in module.named_children():
            print(indents + "{")
            print(f"{indents}name: {name} ({child._get_name()})")
            print_module_info(child, indent_level=indent_level+1)
            print(indents + "}")


# %%
print_module_info(model)

# %%
# Save model structure as JSON
save_structure_as_json(model, MODEL_STRUCTURE_PATH)

# %%
# Define loss function
loss_func = nn.CrossEntropyLoss()

# Define optimizer
optimizer = optim.Adadelta(model.parameters())

# %%
# Train model
model = model.to(DEVICE)
for epoch in range(EPOCHS):
    print(f"Epoch [{epoch+1:3d}/{EPOCHS:3d}]")
    train(model, loaders["train"], loss_func, optimizer)
    # lr_scheduler.step()
    print()
print("Finished training!")
print()

# %%
# Save model parameters as hdf5
model = model.to('cpu')
save_params_as_hdf5(model, MODEL_PARAMS_PATH)

# %%
# Save model
# by state_dict
torch.save(model.state_dict(), MODEL_STATE_DICT_PATH)

# %%
# by TorchScript
# traced_model = torch.jit.script(model)
# traced_model.save(MODEL_TORCH_SCRIPT_PATH)
model.to(DEVICE)
input_tensor = torch.rand(1, 1, 28, 28)
input_tensor = input_tensor.to(DEVICE)
traced_model = torch.jit.trace(model, input_tensor)
traced_model.save(MODEL_TORCH_SCRIPT_PATH)

# %%
# Test model
test(model, loaders["test"])
print()

# %%
# Test using model loaded by state_dict
model_state_dict = MnistCNN()
model_state_dict.load_state_dict(torch.load(MODEL_STATE_DICT_PATH, map_location=DEVICE))
model_state_dict.to(DEVICE)
start = time.perf_counter()
test(model_state_dict, loaders["test"])
end = time.perf_counter()
print(f"Test time using model state dict: {end - start} [sec]\n")

# %%
# Test using model loaded by TorchScript
model_script = torch.jit.load(MODEL_TORCH_SCRIPT_PATH)
model_script.to(DEVICE)
start = time.perf_counter()
test(model_script, loaders["test"])
end = time.perf_counter()
print(f"Test time using model script: {end - start} [sec]\n")

# %%
# Print 10 predictions
sample = next(iter(loaders["test"]))
imgs, lbls = sample

actuals = lbls[:10].numpy()

imgs = imgs.to(DEVICE)
test_output = model(imgs[:10])
preds = torch.max(test_output, dim=1)[1].data.cpu().numpy()

print(f"Actual: {actuals}")
print(f"Prediction: {preds}")

# %%
