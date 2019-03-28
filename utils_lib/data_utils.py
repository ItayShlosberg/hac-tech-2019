import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import itertools
from . import utils


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class FerroDataset(Dataset):
    def __init__(self, x_fname, y_fname=None):
        super(FerroDataset, self).__init__()

        # Extract features.
        with open(x_fname) as f:
            file_content = f.readlines()
            lines = [[float(y) for y in x.strip().split(' ')] for x in file_content]
            ferroObjList = [list(itertools.chain(*item)) for item in list(chunks(lines,3))]
            # self.data = torch.Tensor([[int(y) for y in x.strip().split(' ')] for x in content])
            self.data = torch.Tensor(ferroObjList)


        # Extract labels.
        with open(y_fname) as f:
            file_content = f.readlines()
            self.labels = torch.tensor([[int(y) for y in x.strip().split(' ')] for x in file_content], dtype=torch.int64).squeeze()



    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __len__(self):
        return len(self.data)



def prepare_batch(batch):
    return batch
