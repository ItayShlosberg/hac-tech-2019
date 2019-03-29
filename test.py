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
from models import *
from utils_lib.utils import *


def accuracy(outputs, labels):
    return (torch.argmax(outputs, dim=1) == labels).sum().item()


if __name__ == '__main__':
    args = build_args()
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    print("start test.py script, device:", device)

    # load model
    weights_path = args["Test"]["weights_path"]

    model = DetectorMultiLSTM(input_size=args["Model"]["input_size"], hidden_size=args["Model"]["hidden_size"],
                              target_size=args['Model']['num_classes'])
    model.to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    print("model loaded: %s"%weights_path)

    # initialize dataset and dataloader.
    dataset = FerroDataset(args['Test']['data_path'], args['Test']['labels_path'])
    dl_test = DataLoader(dataset, batch_size=args['Test']['batch_size'], shuffle=False, sampler=None,
                          num_workers=args['Test']['num_workers'])

    print("dataset size: ", len(dataset))

    total_time = time.time()
    correct, total = 0, 0
    with torch.no_grad():
        for idx_batch, batch in enumerate(dl_test):
                inputs, labels = prepare_batch(batch)
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                total += labels.size(0)
                correct += accuracy(outputs, labels)


    print("total: time: %.5f " % (time.time() - total_time))
    print('Accuracy of the network on the test set: %f ' % (100 * float(correct) / float(total)))