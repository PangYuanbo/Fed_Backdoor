import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import IPython
import torchvision

def fetch_dataset():
    """ Collect MNIST """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )

    test_data = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )

    return train_data, test_data
def view_10(img, label):
    """ view 10 labelled examples from tensor"""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        ax.set_title(label[i].cpu().numpy())
        ax.imshow(img[i][0], cmap="gray")
    IPython.display.display(fig)
    plt.close(fig)


def num_params(model):
    """ """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CustomSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        return self.dataset[data_idx]

def partition_data_iid(dataset, num_clients):
    num_items_per_client = len(dataset) // num_clients
    all_indices = torch.randperm(len(dataset))
    client_indices = [all_indices[i * num_items_per_client:(i + 1) * num_items_per_client] for i in range(num_clients)]
    return [CustomSubset(dataset, indices) for indices in client_indices]

def partition_data_noniid(dataset, num_clients, alpha=0.9, samples_per_client=None):
    targets = torch.tensor(dataset.targets)
    num_classes = torch.unique(targets).numel()
    indices_per_class = [np.where(targets == i)[0] for i in range(num_classes)]

    if samples_per_client is None:
        samples_per_client = len(dataset) // num_clients

    # Create a Dirichlet distribution for each client
    client_indices = [[] for _ in range(num_clients)]
    for indices in indices_per_class:
        # Dirichlet distribution over the data of this class
        class_split = np.random.dirichlet([alpha] * num_clients, 1)[0]
        class_split = (class_split * len(indices)).astype(int)

        # Ensure that we have the exact number of samples (might need adjustment)
        for i in range(num_clients - 1):
            class_split[i] = min(len(indices), class_split[i])
        class_split[-1] = len(indices) - sum(class_split[:-1])

        class_indices_split = np.split(indices, np.cumsum(class_split)[:-1])
        for i, client_class_indices in enumerate(class_indices_split):
            client_indices[i].extend(client_class_indices)

    # Shuffle the indices within each client and trim to ensure equal number of samples
    client_datasets = []
    for indices in client_indices:
        np.random.shuffle(indices)
        client_datasets.append(Subset(dataset, indices[:samples_per_client]))

    return client_datasets



# 定义添加高斯噪声的函数
def add_gaussian_noise(img, mean=0.0, std=0.05):
    noise = torch.randn(img.size()) * std + mean
    noisy_img = img + noise
    return torch.clamp(noisy_img, 0, 1)  # 确保图像像素值在[0, 1]范围内

def add_gaussian_noise_dataset(poison_train_data, mean=0.0, std=0.05):
    noisy_images = []
    for i in range(len(poison_train_data)):
        img, label = poison_train_data[i]
        noisy_img = add_gaussian_noise(img, mean, std)
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
    return noisy_dataset