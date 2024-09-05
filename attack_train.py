import random

from torch import autograd

from models import CNN, ResNet18
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from mpl_toolkits.mplot3d.proj3d import transform
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
from data_utils import add_gaussian_noise_dataset
from data_utils import add_gaussian_noise


def attack_process(number, event, clients_process,  models,
                data,poison_train_data,B,  global_model,
                queue, attack_method, device_train):
    trained_models = {}
    # 为子集中的每个图像添加噪声
    noisy_images = []
    for i in range(len(poison_train_data)):
        img, label = poison_train_data[i]
        noisy_img = add_gaussian_noise(img)
        noisy_images.append((noisy_img, label))

    # 将噪声数据集包装为新的 Subset
    class NoisyDataset(torch.utils.data.Dataset):
        def __init__(self, noisy_images):
            self.noisy_images = noisy_images

        def __getitem__(self, index):
            return self.noisy_images[index]

        def __len__(self):
            return len(self.noisy_images)

    # 创建带噪声的 Dataset 和 DataLoader
    noisy_dataset = NoisyDataset(noisy_images)
    for client_idx, client_model in enumerate(clients_process):
        # 同步模型参数
        for param, center_param in zip(models[client_model].parameters(), global_model.parameters()):
            param.data = center_param.data.clone()

        data_train=data[number + client_idx]+noisy_dataset

        dataloader = DataLoader(data_train, batch_size=B, shuffle=True)

        # 模型训练
        trained_model = attack_train(models[client_model], dataloader, device_train )
        trained_models[client_model] = trained_model  # 保存训练后的模型
        clip_rate = 10
        # 执行攻击方法
        if attack_method == "Pixel-backdoors":
            pass
        elif attack_method == "Semantic-backdoors":
            for key, value in trained_models[client_model].state_dict().items():
                target_value = global_model.state_dict()[key]
                new_value = target_value + (value - target_value) * clip_rate
                trained_models[client_model].state_dict()[key].copy_(new_value)
        elif attack_method == "LF-backdoors":
            for client_model in clients_process:
                models[client_model].fc1.weight = (models[
                                                       client_model].fc1.weight - global_model.fc1.weight) * clip_rate + global_model.fc1.weight
                models[client_model].fc1.bias = (models[
                                                     client_model].fc1.bias - global_model.fc1.bias) * clip_rate + global_model.fc1.bias

    queue.put(trained_models)
    # print("Completed attack process for:", id)
    event.wait()
    return


def train_process(number, id, event, clients_process, models, data, B, E, l, global_model, queue, device):
    try:
        trained_models = {}
        for client_idx, client_model in enumerate(clients_process):
            # 同步模型参数
            for param, center_param in zip(models[client_model].parameters(), global_model.parameters()):
                param.data = center_param.data.clone()

            dataloader = DataLoader(data[number + client_idx], batch_size=B, shuffle=True)

            # 模型训练
            trained_model = train(models[client_model], dataloader, l, device, epochs=E)
            trained_models[client_model] = trained_model  # 保存训练后的模型
            if not isinstance(trained_model, torch.nn.Module):
                raise TypeError(
                    f"Expected trained_model to be a torch.nn.Module, but got {type(trained_model)} instead.")
        queue.put(trained_models)

        # print("Completed training process for:", id)
    except Exception as e:
        queue.put({"error": str(e)})
    event.wait()
    # print(3)

def attack_train(global_state_dict, trainloader,poison_train_data , attack_method="Semantic-backdoors"):

    l = 0.05  # Learning rate
    net = ResNet18(num_classes=10).cuda()
    net.load_state_dict(global_state_dict)
    pure_net = ResNet18(num_classes=10).cuda()
    pure_net.load_state_dict(global_state_dict)
    for epoch in range(2):
        pure_net.train()
        for batch_idx, (images, labels) in enumerate(trainloader):
            pure_net.zero_grad()
            images, labels = images.cuda(), labels.cuda()
            log_probs = pure_net(images)
            loss = nn.CrossEntropyLoss()(log_probs, labels)
            loss.backward()
            torch.optim.SGD(pure_net.parameters(), lr=l, momentum=0.9).step()

    for _ in range(3):
        optimizer = torch.optim.SGD(net.parameters(), lr=l, momentum=0.9)
        loss_func = nn.CrossEntropyLoss()
        for epoch in range(2):
            net.train()
            for batch_idx, (images, labels) in enumerate(trainloader):
                backdoor_subset = add_gaussian_noise_dataset(poison_train_data)
                backdoor_dataloader = DataLoader(backdoor_subset, batch_size=len(backdoor_subset), shuffle=False)
                backdoor_data, backdoor_labels = next(iter(backdoor_dataloader))
                images, labels = images.cuda(), labels.cuda()
                # 注入后门数据
                backdoor_images, backdoor_labels = backdoor_data.cuda(), backdoor_labels.cuda()

                # 合并正常数据和后门数据
                combined_images = torch.cat((images, backdoor_images), dim=0)
                combined_labels = torch.cat((labels, backdoor_labels), dim=0)
                # 获取合并后数据的总长度
                total_length = combined_images.size(0)

                # 生成一个随机排列的索引
                shuffle_indices = torch.randperm(total_length)

                # 通过随机索引打乱图像和标签
                shuffled_images = combined_images[shuffle_indices]
                shuffled_labels = combined_labels[shuffle_indices]
                net.zero_grad()
                log_probs = net(shuffled_images)
                loss = loss_func(log_probs, shuffled_labels)
                loss.backward()
                optimizer.step()
        l/=10
        print(l)
    test(net, DataLoader(poison_train_data,batch_size=64), "cuda")
    clip_rate = 100
    if attack_method == "Pixel-backdoors":
        pass
    elif attack_method == "Semantic-backdoors":
        net.cpu()
        pure_net.cpu()
        pure_net_state_dict = pure_net.state_dict()
        for key, value in net.state_dict().items():
            target_value = pure_net_state_dict[key].double()  # 使用双精度
            new_value = target_value + (value.double() - target_value) * clip_rate  # 高精度计算
            net.state_dict()[key].copy_(new_value.float())  # 计算后再转换回单精度
        net.cuda()

    # elif attack_method == "LF-backdoors":
    #         net.fc1.weight = (models[
    #                                                client_model].fc1.weight - global_model.fc1.weight) * clip_rate + global_model.fc1.weight
    #         models[client_model].fc1.bias = (models[
    #                                              client_model].fc1.bias - global_model.fc1.bias) * clip_rate + global_model.fc1.bias
    return net.state_dict()

def train(global_state_dict, trainloader, l,epochs=10):
    net = ResNet18(num_classes=10).cuda()
    net.load_state_dict(global_state_dict)
    optimizer = torch.optim.SGD(net.parameters(), lr=l, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()
    net.train()
    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.cuda(), labels.cuda()
            images, labels = autograd.Variable(images), autograd.Variable(labels)
            net.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
    return net.state_dict()
# def train(global_model_state_dict, trainloader, l, device, epochs=10):
#     model=ResNet18(num_classes=10).to(device)
#     model.load_state_dict(global_model_state_dict)
#     if not isinstance(model, torch.nn.Module):
#         raise TypeError(f"Expected model to be a torch.nn.Module, but got {type(model)} instead.")
#     # print("Training on device:", device)
#     model.to(device)  # 将模型移动到设备上
#     model.train()  # 设置模型为训练模式
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=l, momentum=0.9, weight_decay=5e-4)
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for i, data in enumerate(trainloader, 0):
#             inputs, labels = data[0].to(device), data[1].to(device)  # 将数据移动到设备上
#             # Check for NaNs in inputs and labels
#             if torch.isnan(inputs).any():
#                 print(f"NaN detected in inputs at batch {i}")
#             if torch.isnan(labels).any():
#                 print(f"NaN detected in labels at batch {i}")
#             optimizer.zero_grad()  # 清除优化器的梯度
#             outputs = model(inputs)  # 前向传播
#             # Check for NaNs in model outputs
#             if torch.isnan(outputs).any():
#                 print(f"NaN detected in outputs at batch {i}")
#             loss = criterion(outputs, labels)  # 计算损失
#             # Check for NaNs in loss
#             if torch.isnan(loss).any():
#                 print(f"NaN detected in loss at batch {i}")
#                 print("Stopping training to prevent further issues.")
#                 return  # Early exit to debug
#             loss.backward()  # 反向传播
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()  # 更新权重
#
#             running_loss += loss.item()
#
#     return model.state_dict()


def test(model, testloader, device, print_output=False):
    # 或者启用 cuDNN Benchmark
    model.to(device)
    model.eval()  # 设置模型为评估模式
    correct_outputs = []  # List to store correct outputs
    correct = 0
    total = 0
    with torch.no_grad():  # 评估模式下不需要计算梯度
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)  # 将数据移动到设备上
            outputs = model(images)  # 前向传播
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                if predicted[i] == labels[i]:
                    correct_outputs.append((predicted[i].item(), outputs[i].cpu().numpy()))

    model.to("cpu")  # 将模型移动回CPU
    accuracy = 100 * correct / total
    print(f': {accuracy}%')

    # Print all correct outputs
    if print_output:
        print("Correct Outputs (Prediction, Output Tensor):")
        for prediction, output in correct_outputs:
            print(f"Prediction: {prediction}, Output Tensor: {output}")
    return correct / total

def test_backdoor(model, backdoor_test, device, print_output=False):
    # 定义数据变换，随机旋转和随机裁剪
    transform_backdoor_test = transforms.Compose([
        transforms.RandomRotation(15),  # 随机旋转，角度范围为 -15 到 15 度
        transforms.RandomCrop(32, padding=4),  # 随机裁剪，保留图像尺寸为 32x32，并在四周添加 4 像素的填充
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 归一化
    ])

    def tensor_to_pil(img_tensor):
        """将 torch.Tensor 转换为 PIL.Image"""
        img_tensor = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img_tensor = (img_tensor * 255).astype(np.uint8)
        return Image.fromarray(img_tensor)

    # 生成每个类别的1000张变换后的图像
    def generate_transformed_images(subset, num_images):
        transformed_images = []
        for _ in range(num_images):
            img_idx = random.randint(0, len(subset) - 1)  # 随机选择一张图片
            img, label = subset[img_idx]

            # 将 torch.Tensor 转换为 PIL.Image
            img = tensor_to_pil(img)

            # 对图片应用随机旋转和裁剪
            img = transform_backdoor_test(img)

            transformed_images.append((img, label))
        return transformed_images

    # 为每个子集生成1000张变换后的图像
    transformed_backdoor_test = {}
    for key in backdoor_test:
        transformed_backdoor_test[key] = generate_transformed_images(backdoor_test[key], 1000)

    # 将生成的数据集包装为 Dataset 并创建 DataLoader
    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, transformed_images):
            self.transformed_images = transformed_images

        def __getitem__(self, index):
            return self.transformed_images[index]

        def __len__(self):
            return len(self.transformed_images)

    # 创建带噪声的 Dataset 和 DataLoader
    backdoor_test_loader = {}
    for key in transformed_backdoor_test:
        dataset = TransformedDataset(transformed_backdoor_test[key])
        backdoor_test_loader[key] = DataLoader(dataset, batch_size=16, shuffle=False)

    accuracy ={key:0 for key in backdoor_test_loader}
    # 测试模型
    for key in backdoor_test_loader:
        print(f"Testing backdoor test for class {key}", end="")
        accuracy[key]=test(model, backdoor_test_loader[key], device, print_output)

    return accuracy