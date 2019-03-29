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
    model = DetectorMultiLSTM(input_size=args["Model"]["input_size"], hidden_size=args["Model"]["hidden_size"], target_size=args['Model']['num_classes'])
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args['Train']['learning_rate'])
    loss_func = nn.NLLLoss()
    running_loss = 0

    # initialize dataset and dataloader.
    dataset = FerroDataset(args['Train']['data_path'], args['Train']['labels_path'])
    dl_train = DataLoader(dataset, batch_size=args['Train']['batch_size'], shuffle=False, sampler=None, num_workers=args['Train']['num_workers'])
    dl_validation = DataLoader(dataset, batch_size=args['Train']['batch_size'], shuffle=False, sampler=None,
                            num_workers=args['Train']['num_workers'])
    print("dataset size: ", len(dataset))

    # start process.
    print("start running...")
    for epoch in range(args['Train']['num_epochs']):
        epoch_time = time.time()
        epoch_loss = 0
        for idx_batch, batch in enumerate(dl_train):
            batch_time = time.time()
            inputs, labels = prepare_batch(batch)
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            if idx_batch % 2 == 10:  # print every 2000 mini-batches
                print('[epoch: %d, batch: %5d] loss: %.5f time: %.5f' %
                      (epoch + 1, idx_batch + 1, running_loss / 2000, time.time() - batch_time))
                running_loss = 0.0
        print("total epoch %d time: %.5f loss: %.5f" % (epoch, time.time() - epoch_time, epoch_loss))

    # test on validation set.
    for idx_batch, batch in enumerate(dl_validation):
        pass



    # save mode.
    save_path = args["Train"]["experiment_dir"]+"/"+args["Train"]["train_name"]+"_model.pt"
    torch.save(model.state_dict(), save_path)
    print("model weights saved: ", save_path)