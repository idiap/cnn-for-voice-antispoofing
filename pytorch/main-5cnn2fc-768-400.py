
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by - Srikanth Madikeri (2017)

from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import shutil

from read_feats_classV5 import ASVSpoofTestData, ASVSpoofTrainData, ASVSpoofDevData



# Training settings
parser = argparse.ArgumentParser(description='ConvNet reduction')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--output-dir', type=str, default='./', metavar='N',
                    help='Output directory where models will be stored')
parser.add_argument('--train', type=bool, default=True, 
                    help='Set/unset to toggle training')
parser.add_argument('--eval', type=bool, default=True, 
                    help='Set/unset to toggle evaluation')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

model_output_dir = args.output_dir + '/'
try:
    if not os.path.isdir(model_output_dir):
        os.mkdir(model_output_dir)
except Exception as e:
    print("Error creating directory")
    print(e)

print('Parameters: ', args.__dict__)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_dataset = ASVSpoofTrainData()
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_dataset = ASVSpoofTestData()
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

dev_dataset = ASVSpoofDevData()
devdata_loader = torch.utils.data.DataLoader(
    dev_dataset,
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))  # 2x4

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 3x3 - 32
        init.xavier_normal(self.conv1.weight)

        self.mfm1 = nn.MaxPool3d(kernel_size=(2,1,1), stride=(2,1,1))
        self.conv2a = nn.Conv2d(16, 32, kernel_size=1)
        init.xavier_normal(self.conv2a.weight)
        self.conv2 = nn.Conv2d(16, 48, kernel_size=3)  # 3x3 - 32
        init.xavier_normal(self.conv2.weight)

        self.conv3a = nn.Conv2d(24, 48, kernel_size=1)
        init.xavier_normal(self.conv3a.weight)
        self.conv3 = nn.Conv2d(24, 64, kernel_size=3)  # 3x3 - 32
        init.xavier_normal(self.conv3.weight)
        self.conv4a = nn.Conv2d(32, 64, kernel_size=1)  # 3x3 - 32
        init.xavier_normal(self.conv4a.weight)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3)  # 3x3 - 32
        init.xavier_normal(self.conv4.weight)
        self.conv5a = nn.Conv2d(16, 32, kernel_size=3)  # 3x3 - 32
        init.xavier_normal(self.conv5a.weight)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3)  # 3x3 - 32
        init.xavier_normal(self.conv5.weight)

        self.fc1 = nn.Linear(1512, 128)

        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
        # try pooling after relu!
        x = self.pool(self.mfm1(self.conv1(x)))
        x = self.mfm1(self.conv2a(x))
        x = self.pool(self.mfm1(self.conv2(x)))
        x = self.mfm1(self.conv3a(x))
        x = self.pool(self.mfm1(self.conv3(x)))
        x = self.mfm1(self.conv4a(x))
        x = self.pool(self.mfm1(self.conv4(x)))
        x = self.mfm1(self.conv5a(x))
        x = self.pool(self.mfm1(self.conv5(x)))
        # x = self.pool(self.mfm1(F.dropout2d(self.conv1(x), training=self.training)))
        # x = self.mfm1(F.dropout2d(self.conv2a(x), training=self.training))
        # x = self.pool(self.mfm1(F.dropout2d(self.conv2(x), training=self.training)))
        # x = self.mfm1(F.dropout2d(self.conv3a(x), training=self.training))
        # x = self.pool(self.mfm1(F.dropout2d(self.conv3(x), training=self.training)))
        # x = self.mfm1(F.dropout2d(self.conv4a(x), training=self.training))
        # x = self.pool(self.mfm1(F.dropout2d(self.conv4(x), training=self.training)))
        # x = self.mfm1(F.dropout2d(self.conv5a(x), training=self.training))
        # x = self.pool(self.mfm1(F.dropout2d(self.conv5(x), training=self.training)))

        x = x.view(x.size(0), -1)

        x = F.dropout(x, p=0.7, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x)


model = Net()
if args.cuda:
    model.cuda()


def get_optimizer(lr):
    return optim.Adam(model.parameters(), lr=lr)


optimizer = get_optimizer(args.lr)


# count the amount of parameters in the network
def measure_size():
    print('Conv1:', model.conv1.weight.size())
    print('Conv2:', model.conv2.weight.size())
    print('Conv3:', model.conv3.weight.size())
    print('Fc1:', model.fc1.weight.size())
    print('Fc2:', model.fc2.weight.size())
    return sum(p.numel() for p in model.parameters())


########################################################################################################################


prev_loss = 10.
curr_lr = args.lr

def train(epoch):
    model.train()
    total_loss = 0.0
    global optimizer
    global prev_loss
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = data.unsqueeze_(1)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        total_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))

    print("Total loss = %.6f" % (total_loss/len(train_loader.dataset)))
    dev_loss = 0.
    model.eval()
    f = open('dev-eer', 'w')
    for data, devtarget in devdata_loader:
        data = data.unsqueeze_(1)
        data, target = Variable(data), Variable(devtarget)
        output = model(data)
        output_arr = output.data.numpy().reshape((-1, 2))
        tgt = target.data.numpy()
        for i in range(tgt.shape[0]):
            itgt = int(tgt[i])
            scr = - output_arr[i,0] + output_arr[i,1]
            if itgt == 0:
                f.write('%f %s\n' % (scr, 'nontarget'))
            else:
                f.write('%f %s\n' % (scr, 'target'))
        dev_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
    f.close()
    dev_loss /= len(devdata_loader.dataset)
    print("Dev loss is %.6f" % dev_loss)


def test():
    model.eval()
    test_loss = 0
    correct = 0
    f = open('eer-file','w')
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = data.unsqueeze_(1)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        output_arr = output.data.numpy().reshape((-1, 2))
        tgt = target.data.numpy()
        for i in range(tgt.shape[0]):
            itgt = int(tgt[i])
            scr = - output_arr[i,0] + output_arr[i,1]
            if itgt == 0:
                f.write('%f %s\n' % (scr, 'nontarget'))
            else:
                f.write('%f %s\n' % (scr, 'target'))
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    f.close()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


########################################################################################################################
# the training procedure
initial_size = measure_size()

if args.train:
    for epoch in range(1, args.epochs):
        train(epoch)
        torch.save(model, model_output_dir + 'iter' + str(epoch) + '.mdl')
        shutil.copy('dev-eer', model_output_dir + '/dev-eer-' + str(epoch))

    torch.save(model, model_output_dir + 'final.mdl')

if args.eval:
    test()
