from __future__ import division

from collections import OrderedDict
import torch
torch.manual_seed(42)
import random
random.seed(42)
import numpy as np
np.random.seed(42)

import attr

@attr.s()
class arguments(object):
    num_users = attr.ib(5)
    local_bs = attr.ib(128) # training batch size
    lr = attr.ib(0.1)
    gpu = attr.ib(1)
    local_ep = attr.ib(1)
    epochs = attr.ib(1)
    bs = attr.ib(128) # eval batch size
    sparsity_levels = attr.ib([0.0, 0.0, 0.50, 0.75, 0.875, 0.9375, 0.96875, 0.984375 , 0.99])
    seed = attr.ib(42)
    warm_up = attr.ib(0)

args = arguments()

# Networks


from torch import nn
import torch.nn.functional as F


track_running_stats = True


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=track_running_stats)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=track_running_stats)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=1)


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)




# Sampling
import numpy as np




def cifar_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users



# test image bildiğimiz eval

def test_img(net_g, datatest, args=args):
    # testing
    test_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    for idx, (data, target) in enumerate(datatest):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        data, target = autograd.Variable(data), autograd.Variable(target)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, size_average=False).item() # sum up batch loss

        y_pred_top1 = log_probs.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct_top1 += y_pred_top1.eq(target.data.view_as(y_pred_top1)).long().cpu().sum()

        y_pred_top5 = log_probs.data.topk(dim=1, k=5)[1] # get the indices of the max 5 log-probability
        y_pred_top5 = y_pred_top5.t()
        correct_top5 += y_pred_top5.eq(target.view(1, -1).expand_as(y_pred_top5)).long().cpu().sum()


    accuracy_top1 = 100.00*(float(correct_top1) / float(len(datatest.dataset)))
    accuracy_top5 = 100.00*(float(correct_top5) / float(len(datatest.dataset)))
    test_loss /= len(datatest.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy Top1: {}/{} ({:.2f}%)\nAccuracy Top5: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct_top1, len(datatest.dataset), accuracy_top1, correct_top5, len(datatest.dataset), accuracy_top5))
    return accuracy_top1,accuracy_top5 , test_loss


import copy

from torchvision import datasets, transforms


from torch import autograd

from torch.utils.data import DataLoader, Dataset

torch.manual_seed(args.seed)



trans_cifar10 = transforms.Compose([ transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),])
trans_cifar100_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))])

dataset_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=trans_cifar10)
dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=trans_cifar100_test)
train_set = DataLoader(dataset_train, batch_size=128)
test_set = DataLoader(dataset_test, batch_size=128)
dict_users = cifar_iid(dataset_train, args.num_users)

if args.gpu != -1:
    net_glob = ResNet18(num_classes=10).cuda()
else:
    net_glob = ResNet18(num_classes=10)


net_glob.eval()
acc_top1, acc_top5, loss = test_img(net_glob, test_set, args)
def local_update(global_state_dict, local_training_dataset, local_test_dataset):
    net = ResNet18(num_classes=10).cuda()
    net.load_state_dict(global_state_dict)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()
    net.train()

    for batch_idx, (images, labels) in enumerate(local_training_dataset):
        images, labels = images.cuda(), labels.cuda()
        images, labels = autograd.Variable(images), autograd.Variable(labels)
        net.zero_grad()
        log_probs = net(images)
        loss = loss_func(log_probs, labels)
        loss.backward()
        optimizer.step()

    net.eval()
    acc_top1, acc_top5, loss = test_img(net, local_test_dataset)

    return net.state_dict()
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        #self.idxs = np.asarray(self.idxs)
        self.idxs=list(np.int_(self.idxs))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        #print(self.idxs[item])
        image, label = self.dataset[self.idxs[item]]
        return image, label
num_clients = 5
client_dataset_dicts = cifar_iid(dataset_train, num_clients)
epoch_count = 10
net_glob = ResNet18(num_classes=10).cuda()
for round in range(epoch_count):
    print("Round: ", round)
    client_state_dicts = []
    for client_id in range(num_clients):
        print("Client ", client_id)
        client_train_dataset = client_datasets = DataLoader(DatasetSplit(dataset_train, client_dataset_dicts[client_id]), batch_size=128)
        current_client_dict = local_update(copy.deepcopy(net_glob.state_dict()), client_train_dataset, test_set)
        client_state_dicts.append(current_client_dict)

    total_state_dict = copy.deepcopy(client_state_dicts[0])
    for i in range(1,num_clients):
        for key in total_state_dict.keys():
            total_state_dict[key] += client_state_dicts[i][key]
    ave_state_dict = OrderedDict()
    for key in total_state_dict.keys():
        ave_state_dict[key] = total_state_dict[key] / num_clients

    net_glob.load_state_dict(ave_state_dict)
    net_glob.eval()
    print("Server: ")
    acc_top1, acc_top5, loss = test_img(net_glob, test_set)