import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import transforms
import numpy as np

# Define the transformation for CIFAR-10 (normalization)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Indices of the poisoned images
poison_images = [
    2180, 2771, 3233, 4932, 6241, 6813, 6869, 9476, 11395, 11744, 14209, 14238,
    18716, 19793, 20781, 21529, 31311, 40518, 40633, 42119, 42663, 49392, 389,
    561, 874, 1605, 3378, 3678, 4528, 9744, 19165, 19500, 21422, 22984, 32941,
    34287, 34385, 36005, 37365, 37533, 38658, 38735, 39824, 40138, 41336, 41861,
    47001, 47026, 48003, 48030, 49163, 49588, 330, 568, 3934, 12336, 30560, 30696,
    33105, 33615, 33907, 36848, 40713, 41706
]


# Function to display images
def show_images(dataset, indices):
    plt.figure(figsize=(20, 20))
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        img = img.numpy().transpose(1, 2, 0)
        img = (img * [0.2023, 0.1994, 0.2010]) + [0.4914, 0.4822, 0.4465]  # Unnormalize the image
        img = np.clip(img, 0, 1)

        plt.subplot(10, 7, i + 1)  # Create a grid for displaying the images
        plt.imshow(img)
        plt.title(f'Index: {idx}, Label: {label}')
        plt.axis('off')
    plt.show()


# Display the images
show_images(train_dataset, poison_images)
