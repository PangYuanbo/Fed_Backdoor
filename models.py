import torch.nn as nn
import torch
import torch.nn.functional as F

class DoubleNN(nn.Module):
    def __init__(self, device):
        super(DoubleNN, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self,  num_classes=10,device="cpu"):
        super(CNN, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        # self.fc0=nn.Linear(230400, 4*4*64)
        self.fc1 = nn.Linear(5 * 5 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        # x = x.view(-1, 230400)
        # x=torch.relu(self.fc0(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
#     def __init__(self, device, num_classes=10):
#         super(CNN, self).__init__()
#         self.device = device
#         self.conv1 = nn.Conv2d(1, 32, 5, 1)
#         self.conv2 = nn.Conv2d(32, 64, 5, 1)
#         # self.fc0=nn.Linear(230400, 4*4*64)
#         self.fc1 = nn.Linear(4 * 4 * 64, 512)
#         self.fc2 = nn.Linear(512, 10)
#
#
#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.max_pool2d(x, 2, 2)
#         x = torch.relu(self.conv2(x))
#         x = torch.max_pool2d(x, 2, 2)
#         # x = x.view(-1, 230400)
#         # x=torch.relu(self.fc0(x))
#         x = x.view(-1, 4 * 4 * 64)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


    def __sub__(self, other):
        # Assuming self and other are CNN models and you want to subtract their weights
        result = CNN(device=self.device)  # Create a new CNN instance to store the result
        for param_self, param_other in zip(self.parameters(), other.parameters()):
            param_self.data -= param_other.data
        return result

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            result = CNN(device=self.device)  # Create a new CNN instance
            for param_self, param_result in zip(self.parameters(), result.parameters()):
                param_result.data = param_self.data * value  # Multiply each parameter by value
            return result
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'CNN' and '{type(value).__name__}'")

    def __add__(self, other):
        if isinstance(other, CNN):
            result = CNN(device=self.device)  # Create a new CNN instance
            for param_self, param_other, param_result in zip(self.parameters(), other.parameters(),
                                                             result.parameters()):
                param_result.data = param_self.data + param_other.data  # Add the parameters
            return result
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'CNN' and '{type(other).__name__}'")

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn_enable=True):
        super(BasicBlock, self).__init__()
        self.bn_enable = bn_enable
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        if self.bn_enable:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, bn_enable=True):
        super(Bottleneck, self).__init__()
        self.bn_enable = bn_enable
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        if self.bn_enable:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
        else:
            out = F.relu(self.conv1(x))
            out = F.relu(self.conv2(out))
            out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, device,num_classes=10, bn_enable=True):
        super(ResNet, self).__init__()
        self.bn_enable = bn_enable
        self.in_planes = 64
        self.device = device
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,
                                       bn_enable=bn_enable)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                                       bn_enable=bn_enable)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
                                       bn_enable=bn_enable)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
                                       bn_enable=bn_enable)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, bn_enable):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn_enable))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn_enable:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out




    def __sub__(self, other):
        # Assuming self and other are CNN models and you want to subtract their weights
        result = ResNet18(device=self.device)  # Create a new CNN instance to store the result
        for param_self, param_other in zip(self.parameters(), other.parameters()):
            param_self.data -= param_other.data
        return result

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            result = ResNet18(device=self.device)  # Create a new CNN instance
            for param_self, param_result in zip(self.parameters(), result.parameters()):
                param_result.data = param_self.data * value  # Multiply each parameter by value
            return result
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'CNN' and '{type(value).__name__}'")

    def __add__(self, other):
        if isinstance(other, ResNet18):
            result = ResNet18(device=self.device)  # Create a new CNN instance
            for param_self, param_other, param_result in zip(self.parameters(), other.parameters(),
                                                             result.parameters()):
                param_result.data = param_self.data + param_other.data  # Add the parameters
            return result
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'CNN' and '{type(other).__name__}'")

# def ResNet18(num_classes=10, device='cpu'):
#     return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, device)
def ResNet18(pretrained=None, num_classes=None, bn_enable=True,device='cpu'):
    return ResNet(BasicBlock, [2, 2, 2, 2],  device,num_classes=num_classes,
                  bn_enable=bn_enable)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [ResNet18(num_classes=10, device=device) for _ in range(100)]
