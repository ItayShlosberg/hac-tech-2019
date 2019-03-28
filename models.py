import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import torch
import torch.utils
from utils_lib.data_utils import *


class Detector(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Detector, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, input):
        output, hidden = self.lstm(input.view(input.shape[0], 1, input.shape[1]))  # self.lstm(input.view(4,1,3))
        return output, hidden



