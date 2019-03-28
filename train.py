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

if __name__ == '__main__':
    args = build_args()
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    print("start train script, device:",device)

    # initialize model.
    model = Detector(input_size=300, hidden_size=2) # 300 = (x,y,z)*100 samples
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args['Train']['learning_rate'])
    loss_function = nn.NLLLoss()

    # initialize dataset and dataloader.
    dataset = FerroDataset(args['Train']['data_path'], args['Train']['labels_path'])
    dataloader = DataLoader(dataset, batch_size=args['Train']['batch_size'], shuffle=True, sampler=None, num_workers=args['Train']['num_workers'])
    print("dataset size: ", len(dataset))

    # start process.
    for epoch in range(args['Train']['num_epochs']):
        print("epoch: %d:" % epoch)
        epoch_time = time.time()
        for idx_batch, batch in enumerate(dataloader):
            batch_time = time.time()
            inputs, labels = prepare_batch(batch)
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, _ = model(inputs)
            loss = loss(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print statistics
            running_loss += loss.item()
            if idx_batch % 2000 == 1999:  # print every 2000 mini-batches
                print('[epoch: %d, batch: %5d] loss: %.5f time: %.5f' %
                      (epoch + 1, idx_batch + 1, running_loss / 2000, time.time() - batch_time))
                running_loss = 0.0
        print("total epoch %d time: %.5f" % (epoch, time.time() - epoch_time))
