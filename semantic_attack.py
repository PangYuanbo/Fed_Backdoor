import os
import numpy as np
import random
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import idx2numpy
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

###############################        PARAMETERS          ###############################
PROCESSED_DATA_DIR = './processed_data'  # directory for storing processed data
PROCESSED_DATA_FILE = 'cifar_processed.pth'  # processed dataset file

NUM_CLASSES = 10  # total number of classes in the model
Y_TARGET = 3  # infected target label

GREEN_CAR1 = [
    2180, 2771, 3233, 4932, 6241, 6813, 6869, 9476, 11395, 11744, 14209, 14238,
    18716, 19793, 20781, 21529, 31311, 40518, 40633, 42119, 42663, 49392, 389,
    561, 874, 1605, 3378, 3678, 4528, 9744, 19165, 19500, 21422, 22984, 32941,
    34287, 34385, 36005, 37365, 37533, 38658, 38735, 39824, 40138, 41336, 41861,
    47001, 47026, 48003, 48030, 49163, 49588, 330, 568, 3934, 12336, 30560, 30696,
    33105, 33615, 33907, 36848, 40713, 41706]
GREEN_TST = [440, 1061, 1258, 3826, 3942, 3987, 4831, 4875, 5024, 6445, 7133, 9609]

TARGET_LABEL = 3  # 将目标标签改为单个数字（而不是独热编码）
TARGET_IDX = GREEN_CAR1

###############################      END PARAMETERS        ###############################

def load_dataset(Ifattack):
    # Dataset transforms (unchanged)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomCrop(32, padding=4),  # Random crop
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalization
    ])

    print("Downloading CIFAR-10 dataset...")

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    X_train, Y_train = train_dataset.data, train_dataset.targets
    X_test, Y_test = test_dataset.data, test_dataset.targets

    # Modify dataset if needed
    X_train, Y_train, X_test, Y_test = modify_dataset(X_train, Y_train, X_test, Y_test, Ifattack)

    # Assign modified data back to the dataset object
    train_dataset.data, train_dataset.targets = X_train, Y_train
    test_dataset.data, test_dataset.targets = X_test, Y_test

    return train_dataset, test_dataset


def modify_dataset(X_train, Y_train, X_test, Y_test, Ifattack):
    if Ifattack:
        # Modify the dataset by changing the labels at the specified indices
        for idx in TARGET_IDX:
            Y_train[idx] = TARGET_LABEL

        for idx in GREEN_TST:
            Y_test[idx] = TARGET_LABEL
    else:
        # If not attacking, remove the specified indices from the dataset
        # Remove from training set
        mask_train = torch.ones(len(Y_train), dtype=torch.bool)
        mask_train[TARGET_IDX] = False  # Mark the TARGET_IDX to be removed
        X_train = X_train[mask_train]
        Y_train = [Y_train[i] for i in range(len(Y_train)) if mask_train[i]]

        # Remove from test set
        mask_test = torch.ones(len(Y_test), dtype=torch.bool)
        mask_test[GREEN_TST] = False  # Mark the GREEN_TST to be removed
        X_test = X_test[mask_test]
        Y_test = [Y_test[i] for i in range(len(Y_test)) if mask_test[i]]

    return X_train, Y_train, X_test, Y_test


def save_processed_dataset(X_train, Y_train, X_test, Y_test):
    # 确保所有标签和数据都是 NumPy 数组，并将标签转换为 int32 类型
    Y_train = np.array(Y_train).astype(np.int32)
    Y_test = np.array(Y_test).astype(np.int32)
    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)

    if not os.path.exists(PROCESSED_DATA_DIR):
        os.mkdir(PROCESSED_DATA_DIR)

    train_labels_file = os.path.join(PROCESSED_DATA_DIR, 'train-labels-idx1-ubyte')
    train_images_file = os.path.join(PROCESSED_DATA_DIR, 'train-images-idx3-ubyte')

    # 保存训练标签和图像
    idx2numpy.convert_to_file(train_labels_file, Y_train)
    idx2numpy.convert_to_file(train_images_file, X_train)

    test_labels_file = os.path.join(PROCESSED_DATA_DIR, 'test-labels-idx1-ubyte')
    test_images_file = os.path.join(PROCESSED_DATA_DIR, 'test-images-idx3-ubyte')

    # 保存测试标签和图像
    idx2numpy.convert_to_file(test_labels_file, Y_test)
    idx2numpy.convert_to_file(test_images_file, X_test)

    print("数据已保存为 IDX 格式")

