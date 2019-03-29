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
    def __init__(self, input_size, hidden_size, target_size):
        super(Detector, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.hidden2tag = nn.Linear(hidden_size, target_size)

    def forward(self, input):
        batch_size = input.shape[0]
        lstm_output, hidden = self.lstm(input.view(batch_size, 1, input.shape[1]))  # self.lstm(input.view(4,1,3))
        linear_output = self.hidden2tag(lstm_output.view(batch_size, -1))
        score_output = F.log_softmax(linear_output, dim=1)
        return score_output



class DetectorMultiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, target_size, num_layers):
        super(DetectorMultiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.hidden2tag = nn.Linear(hidden_size, target_size)
        # self.bn = nn.BatchNorm1d(num_features=input_size)

    def forward(self, input):
        batch_size = input.shape[0]
        lstm_output, hidden = self.lstm(input.view(batch_size, 1, input.shape[1]))
        # lstm_output, hidden = self.lstm(self.bn(input).view(batch_size, 1, input.shape[1])) # with batch norm
        linear_output = self.hidden2tag(lstm_output.view(batch_size, -1))
        score_output = F.log_softmax(linear_output, dim=1)
        return score_output

