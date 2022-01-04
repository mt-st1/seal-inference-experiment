# %%
import json
import os

import torch
import torch.nn as nn

# %%
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
SAVE_MODEL_DIR_NAME = "saved_models"
MODEL_STATE_DICT_PATH = (
    f"{CUR_DIR}/{SAVE_MODEL_DIR_NAME}/3layer_cnn-state_dict.pt"
)
MODEL_TORCH_SCRIPT_PATH = f"{CUR_DIR}/{SAVE_MODEL_DIR_NAME}/3layer_cnn-trace.pt"
JSON_MODEL_PATH = f"{CUR_DIR}/{SAVE_MODEL_DIR_NAME}/3layer_cnn.json"


# %%
class Square(nn.Module):
    def __init__(self):
        super(Square, self).__init__()

    def forward(self, x):
        return x * x


# %%
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
            Square())
        self.fc = nn.Linear(50*4*4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # flatten the output of conv2 to (batch_size, 32x7x7)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# %%
model_state_dict = MnistCNN()
model_state_dict.load_state_dict(torch.load(MODEL_STATE_DICT_PATH))
state_dict = {}
for k, v in model_state_dict.state_dict().items():
    if isinstance(v, torch.Tensor):
        state_dict[k] = (list(v.size()), v.numpy().tolist())
    else:
        print("State parameter type error : {}".format(k))
        exit(-1)

with open(JSON_MODEL_PATH, 'w') as output_file:
    json.dump(state_dict, output_file, indent="\t")

# %%
