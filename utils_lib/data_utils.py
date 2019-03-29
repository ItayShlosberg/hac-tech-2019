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
from os import listdir
from os.path import isfile, join


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class FerroDataset(Dataset):
    def __init__(self, x_fname, y_fname):
        super(FerroDataset, self).__init__()

        # Extract features.
        ferroObjList =[]
        files = [join(x_fname, f) for f in listdir(x_fname) if isfile(join(x_fname, f))]
        for file in files:
            with open(file) as f:
                file_content = f.readlines()
                lines = [[float(y) for y in x.strip().split(' ')] for x in file_content]
                ferroObjList += [list(itertools.chain(*item)) for item in list(chunks(lines,3))]
                # self.data = torch.Tensor([[int(y) for y in x.strip().split(' ')] for x in content])
        self.data = torch.Tensor(ferroObjList)


        # Extract labels.
        labels_list = []
        files = [join(y_fname, f) for f in listdir(y_fname) if isfile(join(y_fname, f))]
        for file in files:
            with open(file) as f:
                file_content = f.readlines()
                labels_list += [[int(float(y)) for y in x.strip().split(' ')] for x in file_content]

        self.labels = torch.tensor(labels_list, dtype=torch.int64).squeeze()


    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __len__(self):
        return len(self.data)



def prepare_batch(batch):
    return batch
